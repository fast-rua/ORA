# Remote GPU server and web view

`ora-remote` runs one ORA command on a GPU server over SSH and gives you both
halves of the session here: the Pi terminal UI in this terminal and the
robot's cameras in your browser, over one SSH connection with one forwarded
loopback port. It uses only the Python standard library.

    ora-remote gpu-host --repo /srv/ORA play --env synthetic --provider env
    ora-remote gpu-host --repo /srv/ORA interactive --ui pi --provider PROFILE
    ora-remote gpu-host --repo /srv/ORA attach RUN_ID
    ora-remote gpu-host --repo /srv/ORA watch RUN_ID
    ora-remote gpu-host --repo /srv/ORA --web play ...    the terminal UI and the web page
    ora-remote gpu-host --repo /srv/ORA web [RUN_ID]      only the web page

Everything after the command word goes to `ora` on the server unchanged, and
ORA's launch gates apply there: a prepared RunSpec still needs
`--confirm-launch`. ora-remote never adds it, retries SSH or launches
anything itself.

## What you need

- Here: OpenSSH, a browser and ORA (no simulator). `gpu-host` is anything
  `ssh` accepts, aliases in ~/.ssh/config included.
- There: a checkout set up for the environment (`ora setup --env ...`), its
  Node.js agent core and the provider credentials. `--repo` names it (or set
  ORA_REMOTE_REPO); without either, the server's `ora` on PATH runs.
- SSH port forwarding allowed for you on the server. When the server refuses
  the forward, ora-remote says so; the terminal UI still works, without cameras.

## Detach and come back

Ctrl+D (or /detach), or a dropped connection, leaves the run going on the
server. Run the same command again to attach to it (`--new` starts another).
A run started inside an interactive job step (srun --pty) ends with that step.

## In the browser

    ora web [RUN_ID]                   a live Pi-core run: conversation and cameras
    ora replay RUN_ID                  a finished run, rebuilt from its evidence
    ora replay RUN_ID --export OUT.mp4 the replay as a video (Playwright and ffmpeg)

`ora web setup` builds the page once. A replay plays the waiting at 16x and
each robot motion in real simulated time (`--speed`, `--motion-speed`). On a
real robot cell the page also has the cell's controls, as play's terminal does
(the robot's questions, auto-approve, home, hold, float, release, estop,
diagnose; docs/usage/real-cells.md).

## Security

The world view listens on the server's loopback only and is read-only; other
users of a shared server can connect to that port and see the frames. The web
view sits behind a 0600 unix socket and a one-time launch token, and has no
shell or file route. ora-remote passes no credentials.

## Without ora-remote

    ssh -L 127.0.0.1:8765:127.0.0.1:8765 gpu-host
    ora attach RUN_ID --world-port 8765    then open http://127.0.0.1:8765/live here

More: docs/usage/remote.md and docs/usage/interactive.md.
