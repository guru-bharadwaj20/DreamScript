# Demo — run of show

Phase 17.8 (script) and 17.9 (fallback video). Three minutes, one messy sketch, photograph to
running code, recorded ahead of time in case the live run fails.

## Before the audience arrives (T − 15 min)

| Check | Command / action | Pass when |
| :--- | :--- | :--- |
| model server up and warm | `uvicorn src.serve.api:app --port 8000`, then `curl -F image=@tests/fixtures/flowchart.png localhost:8000/predict` once | second call < 1 s (warm cache) |
| backend serving the app | `DREAMSCRIPT_BUNDLE=./dreamscript-app uvicorn app.backend.main:app --port 3000` | `curl localhost:3000/health?upstream=1` reports the bundle and the upstream |
| phone reaches it over **HTTPS** | a tunnel or reverse proxy with a certificate; the camera refuses plain http on a LAN IP | viewfinder opens |
| app installed on the phone | Add to Home Screen | opens full screen |
| the sketch | a flowchart drawn on paper **in thick marker**: start → read x → decision "x > 0?" → yes: print "positive" / no: print "not positive" → end. One crossed-out box, one wobbly arrow, one label written at a slant | the messiness is deliberate and visible |
| fallback | the recorded video (below) open in a paused tab, and the app's stored *examples* reachable offline | one tap away |

Only flowcharts and state machines work from a photograph (report §7). **Do not demo an ER
diagram or a circuit live**: the router is confidently wrong on types it has never seen.

## The three minutes

| Time | Say | Do |
| :--- | :--- | :--- |
| 0:00–0:20 | "Engineers design on paper and whiteboards. Turning that into code is retyping. DreamScript takes one phone photo of a messy sketch and returns code that runs." | hold up the paper; point at the crossed-out box and the wobbly arrow |
| 0:20–0:45 | "This is an installed web app, so it works on Android and iPhone. The photo goes to our server, because the models are 1.3 GB." | open the app from the home screen, then tap capture |
| 0:45–1:15 | "It straightens the page first, because rotation is what breaks detection. Now each stage reports as it finishes: detect, classify, assemble…" | take the photo; let the stage list tick; read out the diagram type and its confidence |
| 1:15–1:45 | "The boxes it found are drawn back onto my photo. Here is the graph it built. Notice it ignored the crossed-out box." | show the overlay, then the graph view |
| 1:45–2:15 | "It misread this label. I fix it here, the fix is logged as training data, and the code regenerates in milliseconds." | retype one label (there is usually one to fix; if not, fix the slanted one anyway) |
| 2:15–2:40 | "And it runs, in a sandbox on the server, never on my phone." | tap **Run**; show the printed output |
| 2:40–3:00 | "From an accurate graph, 70% of programs pass their behavioural tests. From a photo, the graph is the weak link, and the report says exactly how weak. That is the next piece of work." | back to the paper |

## If something fails

| Failure | Recovery line | Action |
| :--- | :--- | :--- |
| `needs_confirmation` | "It wasn't sure of the type, so it asks instead of guessing. That's intended." | pick *flowchart*; continue |
| stopped at a stage | "Every stage either answers or says why not. Here's why." | show the stage table; retake once with better light |
| no network / server down | "Here's the same run, recorded this morning." | play the video |
| wrong code that runs | "This is the failure mode we measured: 81% of photo-to-code runs don't pass their test yet. The fix is in the graph builder." | show the report table |

## The fallback video (17.9)

Record the three minutes above with the phone's screen recorder, plus a second camera on the
paper, and follow the same script on a successful run. Save it as `demo.mp4`, attach it to the
GitHub release rather than committing it to git, and link it here.

*Recorded:* not yet. This has to be recorded on the real phone against the running server. It
cannot be produced from this repository.
