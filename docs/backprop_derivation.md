# Backpropagation through one layer

Phase 6.2.7. The gradient derivation the viva asks for, done on the network Phase 6.2 actually
trains rather than on a generic two-layer sketch: 6.2.1 selected `(512, 256)`, 6.2.2 selected
GELU, and the loss is `nn.CrossEntropyLoss` with mean reduction.

Every expression below is implemented in [`src/classify/backprop.py`](../src/classify/backprop.py)
in numpy with **no autograd anywhere in the forward or backward pass**, and
[`tests/test_backprop_derivation.py`](../tests/test_backprop_derivation.py) checks it two ways:
against central finite differences, and against `torch.autograd` on the same weights. A
derivation nobody has run beside a reference implementation is a claim; these are the numbers
that make it a tested one.

| activation | max relative error vs. central differences (W1) | (b1) | (W2) | (b2) |
| :--- | ---: | ---: | ---: | ---: |
| relu | 2.7e-08 | 2.1e-07 | 3.0e-08 | 1.3e-09 |
| leaky_relu | 4.2e-08 | 1.7e-07 | 8.6e-09 | 3.0e-09 |
| gelu | 5.2e-08 | 2.0e-08 | 1.6e-08 | 4.8e-09 |
| tanh | 3.5e-07 | 2.6e-08 | 1.9e-07 | 1.2e-09 |

Reproduce with `python -m src.classify.backprop --all`.

---

## 1. Notation

`n` rows in the batch, `d` input features, `h` hidden units, `k` classes. Row-major throughout,
so each **row** of `X` is one diagram and each **row** of `W1` is one input feature — this is
sklearn's and torch's convention and it is why the weight matrices below are `(d, h)` and not
`(h, d)`.

| symbol | shape | meaning |
| :--- | :--- | :--- |
| `X` | `(n, d)` | the scaled feature table — 6.1.4's hybrid columns |
| `W1`, `b1` | `(d, h)`, `(h,)` | hidden layer |
| `z1` | `(n, h)` | hidden pre-activation |
| `a1` | `(n, h)` | hidden activation, `g(z1)` |
| `W2`, `b2` | `(h, k)`, `(k,)` | output layer |
| `z2` | `(n, k)` | logits |
| `p` | `(n, k)` | softmax probabilities |
| `y` | `(n,)` | integer class labels |
| `L` | scalar | mean cross-entropy |

## 2. Forward pass

```
z1 = X W1 + b1
a1 = g(z1)
z2 = a1 W2 + b2
p  = softmax(z2)          p[i,j] = exp(z2[i,j]) / sum_c exp(z2[i,c])
L  = -(1/n) sum_i log p[i, y_i]
```

The bias adds a `(h,)` vector to an `(n, h)` matrix, which broadcasts down the rows: every
example gets the same bias, and that is why the bias gradient in §4 is a sum over examples.

## 3. The output-layer error, and why it collapses

This is the step worth being able to produce on a whiteboard, because it looks like it needs a
`(k, k)` Jacobian per row and turns out not to.

Start from the loss as a function of `p`. Only the true-class entry appears:

```
dL/dp[i,c] = -(1/n) * 1/p[i,y_i]      if c = y_i
             0                        otherwise
```

The softmax Jacobian for row `i` is genuinely a `(k, k)` matrix:

```
dp[i,j]/dz2[i,c] = p[i,j] (delta_jc - p[i,c])
```

Compose them. Write `delta2[i,c] = dL/dz2[i,c]` and sum over `j`:

```
delta2[i,c] = sum_j (dL/dp[i,j]) (dp[i,j]/dz2[i,c])
            = -(1/n) (1/p[i,y_i]) * p[i,y_i] (delta_{y_i,c} - p[i,c])
            = -(1/n) (delta_{y_i,c} - p[i,c])
            =  (1/n) (p[i,c] - delta_{y_i,c})
```

The `p[i,y_i]` cancels exactly. In matrix form:

```
delta2 = (p - onehot(y)) / n                    (n, k)
```

**The `(k, k)` Jacobian never has to be built.** This is why every framework fuses softmax and
cross-entropy into one op — separately they are numerically worse and computationally wasteful.
`softmax_cross_entropy_grad` computes the collapsed form, and one of the tests builds the
uncollapsed `diag(p) - p pᵀ` product explicitly and checks the two agree, so the cancellation
above is verified rather than trusted.

The `1/n` comes from the *mean* in the loss. Dropping it is the most common error in a
hand-written backward pass: the network still trains, because the optimizer's learning rate
absorbs a constant, but every gradient is `n` times too large and a learning-rate sweep like
6.2.4's is then measuring the batch size.

## 4. Output layer

`z2 = a1 W2 + b2`, so with `delta2` in hand both gradients are one matrix product each.

For a single weight `W2[u,v]`, `z2[i,v]` is the only logit it touches, and it enters with
coefficient `a1[i,u]`:

```
dL/dW2[u,v] = sum_i delta2[i,v] a1[i,u]
```

which is exactly a matrix product with `a1` transposed:

```
dL/dW2 = a1ᵀ delta2                             (h, n)(n, k) -> (h, k)  ✔
dL/db2 = sum_i delta2[i, :]                     (k,)
```

The shape check is the whole safety net for this step — `(h, k)` is the only way the two
matrices compose, so a transposed derivation cannot even run.

**The bias gradient is a sum, not a mean.** The `1/n` is already inside `delta2`; averaging again
here would divide the bias gradient by `n` relative to the weights and quietly bias the fit.

## 5. Hidden layer — where the chain rule earns its keep

Propagate the error back through `W2`, then through the nonlinearity.

`a1[i,u]` feeds *every* output unit, so its gradient sums over `v`:

```
dL/da1[i,u] = sum_v delta2[i,v] W2[u,v]         =>   dL/da1 = delta2 W2ᵀ      (n, h)
```

Then through `g`, which is elementwise, so the Jacobian is diagonal and the product is a
Hadamard product rather than a matrix one:

```
delta1 = (delta2 W2ᵀ) ⊙ g'(z1)                  (n, h)
```

and the layer's gradients have the same form as §4 one step down:

```
dL/dW1 = Xᵀ delta1                              (d, n)(n, h) -> (d, h)  ✔
dL/db1 = sum_i delta1[i, :]                     (h,)
```

**`g'` is evaluated at `z1`, not at `a1`.** For ReLU the two happen to give the same answer, which
is exactly what makes this an easy bug to keep: a zero in `a1` could have come from any negative
`z1`, and for tanh and GELU the pre-activation is not recoverable from the output at all. This is
why `forward()` caches `z1` — not as an optimisation, but because the backward pass is wrong
without it.

## 6. The activations, and the one factor that distinguishes them

The derivation above is identical for all four of 6.2.2's activations. The entire difference is
one elementwise factor:

| `g(z)` | `g'(z)` | note |
| :--- | :--- | :--- |
| `max(z, 0)` | `1 if z > 0 else 0` | undefined at 0; torch and this module both pick 0 |
| `max(z, 0.01z)` | `1 if z > 0 else 0.01` | no flat region — 6.2.2's dead-unit count is 0 |
| `tanh(z)` | `1 - tanh²(z)` | → 0 for \|z\| > 3, which is the saturation argument |
| `z Φ(z)` | `Φ(z) + z φ(z)` | product rule; `Φ` the normal CDF, `φ` its density |

GELU is written in its exact `erf` form because that is what `nn.GELU()` computes by default.
The widely quoted `tanh` approximation is a *different function* - it differs from the exact
form by up to 4.7e-04 near |z| = 2.7, far above the 1e-6 the tests hold the gradient to - so using
it here would fail the autograd comparison for a reason that has nothing to do with the
derivation being right or wrong.

The tanh row of the error table is the loosest at 3.5e-07, and that is the finite-difference
check being loose, not the gradient: `tanh` is the only one of the four whose second derivative
is large where the units sit, and central differences carry an `O(ε²·f''')` truncation term.
The `torch.autograd` comparison, which has no truncation error, agrees for all four
activations to **5.6e-17 in float64** - machine precision.

## 7. Deeper networks, and what 6.2.1 actually trains

`(512, 256)` has two hidden layers, so §5 applies twice. The recursion is:

```
delta_{l} = (delta_{l+1} W_{l+1}ᵀ) ⊙ g'(z_l)
dL/dW_l   = a_{l-1}ᵀ delta_l          (with a_0 = X)
dL/db_l   = sum_i delta_l[i, :]
```

Two properties of this recursion are worth stating because they are what the rest of Phase 6.2
is about:

**Every backward step multiplies by `g'`.** With tanh, `g' ≤ 1` and typically well below it, so
the error shrinks geometrically with depth — the vanishing-gradient argument. With ReLU, `g'` is
0 or 1, so nothing shrinks but a unit that is 0 for every row contributes exactly nothing,
forever. 6.2.2 measured that: 9 dead units of 768 on the hybrid table, 29 on the handcrafted.

**Dropout is a mask in both directions.** A unit dropped in the forward pass has its `a1` zeroed,
so its column of `dL/dW2` is zero and it receives no update that step — the mask multiplies the
backward pass exactly as it multiplied the forward one. That is why 6.2.3's dropout at 0.5 slows
convergence (roughly half the network is updated per step) while improving the held-out score.

## 8. Parameter count

The quantity 6.2.1 compares against the corpus size, and which follows directly from the shapes
above — every entry of every `W` and `b` receives one gradient per step:

```
params = (d·h1 + h1) + (h1·h2 + h2) + (h2·k + k)
```

For the hybrid table with `d = 161`, `h = (512, 256)`, `k = 5`:

```
(161·512 + 512) + (512·256 + 256) + (256·5 + 5) = 82,944 + 131,328 + 1,285 = 215,557
```

which is 6.2.1's figure, and 161 parameters per training row — the ratio that motivated 6.2.3.

---

## Viva checklist

Six questions this derivation is designed to answer without notes:

1. **Why is `delta2` just `p - onehot(y)`?** §3 — the softmax Jacobian's `p[i,y_i]` cancels
   against the `1/p[i,y_i]` from `d(log)`.
2. **Where does the `1/n` go?** Into `delta2` once, in §3; nowhere else.
3. **Why is `dL/db` a sum and not a mean?** §4 — the bias broadcasts over rows in the forward
   pass, so its gradient sums over them, and the mean is already applied.
4. **Why `a1ᵀ delta2` and not `delta2ᵀ a1`?** §4 — the shapes `(h, n)(n, k)` are the only ones
   that compose to `W2`'s `(h, k)`.
5. **Why `g'(z1)` and not `g'(a1)`?** §5 — `a1` does not determine `z1` for any of the four
   activations, and for ReLU the coincidence hides the error.
6. **What does the activation change?** §6 — one elementwise factor, and nothing else in the
   whole derivation.
