import { AsyncLocalStorage } from "node:async_hooks";
const scope = new AsyncLocalStorage<true>();
/** Trusted preparation metadata only; never an IPC/renderer flag or neural grant. */
export function withPortableSavedDataRead<T>(operation: () => Promise<T>): Promise<T> { return scope.run(true, operation); }
export function portableSavedDataReadOwned(): boolean { return scope.getStore() === true; }
