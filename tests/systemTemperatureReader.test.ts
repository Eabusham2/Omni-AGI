/** Fixed-namespace sensor fixtures only; never read actual hardware or start a process. */
import { expect, it, vi } from "vitest";
import { readSystemTemperatures, type SystemTemperatureIo } from "../src/main/systemTemperatureReader";

function fixture(values: Record<string, string | Error>, names = ["thermal_zone0", "thermal_zone1", "thermal_zone2"]): SystemTemperatureIo {
  return { entries: vi.fn(async function* () { yield* names; }), read: vi.fn(async (zone, field) => {
    const value = values[`${zone}/${field}`];
    if (value instanceof Error) throw value;
    if (value === undefined) throw Object.assign(new Error("Fixture field absent"), { code: "ENOENT" });
    return value;
  }) };
}

it("reports millidegree thermal observations, genuine zero and partial failures without reading unrelated names", async () => {
  const io = fixture({ "thermal_zone0/temp": "42500\n", "thermal_zone0/type": "x86_pkg_temp\n",
    "thermal_zone1/temp": "0\n", "thermal_zone2/temp": new Error("PRIVATE permission failure") },
    ["cooling_device0", "../../private", "thermal_zone0", "thermal_zone1", "thermal_zone2"]);
  expect(await readSystemTemperatures("linux", undefined, io)).toEqual({ interface: "linux-sysfs-thermal-zones", unit: "celsius",
    readings: [{ sensorId: "thermal_zone0", label: "x86_pkg_temp", celsius: 42.5 }, { sensorId: "thermal_zone1", label: null, celsius: 0 }],
    unavailableSensors: [{ sensorId: "thermal_zone2", reason: "unreadable-or-invalid-reading" }], traversalComplete: true });
  expect(vi.mocked(io.read).mock.calls.every(([zone]) => /^thermal_zone[0-9]+$/.test(zone))).toBe(true);
});

it("never presents invalid or missing values as a zero-temperature observation", async () => {
  const io = fixture({ "thermal_zone0/temp": "not measured", "thermal_zone1/temp": "-273151", "thermal_zone2/temp": "Infinity" });
  await expect(readSystemTemperatures("linux", undefined, io)).rejects.toThrow("did not provide readable temperature observations");
});

it("distinguishes an absent passive adapter or sysfs tree from unreadable sensors", async () => {
  const io = fixture({});
  expect(await readSystemTemperatures("darwin", undefined, io)).toBeUndefined();
  expect(await readSystemTemperatures("win32", undefined, io)).toBeUndefined(); expect(io.entries).not.toHaveBeenCalled();
  io.entries = async function* () { throw Object.assign(new Error("Fixture sysfs absent"), { code: "ENOENT" }); };
  expect(await readSystemTemperatures("linux", undefined, io)).toBeUndefined();
  io.entries = async function* () { throw Object.assign(new Error("Fixture read denied"), { code: "EACCES" }); };
  await expect(readSystemTemperatures("linux", undefined, io)).rejects.toThrow("Fixture read denied");
});

it("stops between sensor reads after cancellation without reading type or the next zone", async () => {
  const abort = new AbortController(), io = fixture({});
  io.read = vi.fn(async () => { abort.abort(); return "40000"; });
  await expect(readSystemTemperatures("linux", abort.signal, io)).rejects.toMatchObject({ name: "AbortError" });
  expect(io.read).toHaveBeenCalledOnce(); expect(io.read).toHaveBeenCalledWith("thermal_zone0", "temp", abort.signal);
});
