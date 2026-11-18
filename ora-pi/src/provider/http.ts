// The providers' HTTP client.
//
// Node's global fetch runs on its bundled undici with a 300 s headersTimeout
// and bodyTimeout. ORA's provider timeout is 900 s by default, and Python's
// httpx applies `Timeout(timeout, connect=min(10.0, timeout))`, so a slow
// first token between 300 s and the provider timeout would be cut short here
// and not in Python. The providers therefore fetch through the pinned undici
// package with Agents built for the attempt's own timeouts, phase by phase as
// httpx applies them:
//
//   connect        connect.timeout = min(10, timeout)  -> ConnectTimeout
//   headers wait   headersTimeout = timeout, from the written request -> ReadTimeout
//   body           AttemptControl's per-chunk inactivity timer (transport.ts) -> ReadTimeout
//
// undici's own body timer is off. A dispatcher from this undici cannot be
// handed to the global fetch (Node's bundled undici rejects it with
// UND_ERR_INVALID_ARG), so this module uses this undici's own fetch as well.
//
// Connections: legacy_v1 is Python's long-lived httpx.Client, so its attempts
// share one pooled Agent per provider; a same_credential_deadline_v1 or
// first_token_deadline_v1 attempt is Python's fresh worker AsyncClient, so it
// gets an Agent of its own that closes once its response has been read.

import { Agent, fetch as undiciFetch } from "undici";

/** The per-attempt fields `post` adds to the RequestInit (other fetches ignore them). */
export interface OraFetchInit extends RequestInit {
  oraReadTimeoutS?: number;
  oraFreshConnection?: boolean;
}

/** The connect timeout httpx would use for this read timeout, in ms. */
export function connectTimeoutMs(timeoutSeconds: number): number {
  return Math.max(1, Math.round(Math.min(10.0, timeoutSeconds) * 1000));
}

export function agentOptions(timeoutSeconds: number): { headersTimeout: number; bodyTimeout: number; connect: { timeout: number } } {
  return {
    headersTimeout: Math.max(1, Math.round(timeoutSeconds * 1000)), bodyTimeout: 0,
    connect: { timeout: connectTimeoutMs(timeoutSeconds) },
  };
}

/** The default `fetchImpl` of a provider whose timeout is `timeoutSeconds`. */
export function providerFetch(timeoutSeconds: number): typeof fetch {
  let pooled: Agent | null = null;
  return (async (input: any, init: OraFetchInit = {}) => {
    const { oraReadTimeoutS, oraFreshConnection, ...rest } = init;
    const read = oraReadTimeoutS ?? timeoutSeconds;
    if (!oraFreshConnection && read === timeoutSeconds) {
      pooled ??= new Agent(agentOptions(timeoutSeconds));
      return undiciFetch(input, { ...(rest as any), dispatcher: pooled });
    }
    const agent = new Agent(agentOptions(read));
    try {
      const response = await undiciFetch(input, { ...(rest as any), dispatcher: agent });
      // close() lets the in-flight body finish (or be cancelled), then releases the socket.
      agent.close().catch(() => undefined);
      return response;
    } catch (error) {
      agent.destroy().catch(() => undefined);
      throw error;
    }
  }) as unknown as typeof fetch;
}
