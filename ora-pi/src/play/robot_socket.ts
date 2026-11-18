// The run's robot socket (ora.robot-socket.v1, ora/agent/pi/robot_socket.py), as a codemode
// script's robot calls reach it: the same path `ora-robot` takes from the agent's shell, so the
// Supervisor admits and runs each call in the turn and returns its receipt and new observation.

import { createConnection } from "node:net";

type Json = any;

let requests = 0;

/** One request on its own connection: its `result`, or `{ok: false, error}` for one the socket refused. */
export function robotSocketCall(path: string, method: string, params: Json, origin: Json): Promise<Json> {
  return new Promise((resolve, reject) => {
    const id = `codemode-${++requests}`;
    const socket = createConnection({ path });
    let buffer = "";
    socket.setEncoding("utf8");
    socket.on("connect", () => socket.write(`${JSON.stringify({ id, method, params, origin })}\n`));
    socket.on("data", (chunk: string) => {
      buffer += chunk;
      const newline = buffer.indexOf("\n");
      if (newline < 0) return;
      socket.end();
      try {
        const message = JSON.parse(buffer.slice(0, newline));
        resolve(message.error ? { ok: false, error: message.error } : message.result);
      } catch (error) {
        reject(error);
      }
    });
    socket.on("error", reject);
    socket.on("close", () => reject(new Error("the robot socket closed before it answered")));
  });
}
