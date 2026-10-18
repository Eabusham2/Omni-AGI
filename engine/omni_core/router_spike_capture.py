"""Opt-in, resource-leased observation of actual LIF hard-spike channels.

The packed transient ring is not learned memory and is never exported as
neural state. No semantic-neuron/contact identity is guessed from activity.
"""
from __future__ import annotations

import math
import re
import struct
import threading
import time
import uuid
import weakref
from typing import Any, Mapping

from .router_activity_inspection import (
    RouterContactSnapshot, capture_router_contacts, empty_router_contact_page,
    empty_router_transmission_page,
)


_HEADER = struct.Struct("<QQdII")
_RING_BYTES = 64 * 1024
_METADATA_ALLOWANCE = 8 * 1024
_VIEWPORT_CHANNELS = 256
_FRAME_PAGE = 32
_SAFE_INTEGER = (1 << 53) - 1
_EPOCH = re.compile(r"[0-9a-f]{32}\Z")


def _integer(value, name, default=0, *, positive=False):
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or not (1 if positive else 0) <= value <= _SAFE_INTEGER:
        raise ValueError("invalid router spike " + name)
    return value


def _query(raw):
    if not isinstance(raw, Mapping) or not isinstance(raw.get("enabled"), bool):
        raise ValueError("invalid router spike request")
    capture_id = raw.get("captureId")
    if not isinstance(capture_id, str) or not capture_id or len(capture_id) > 128:
        raise ValueError("invalid router spike capture identity")
    epoch = raw.get("epoch")
    if epoch is not None and (not isinstance(epoch, str) or not _EPOCH.fullmatch(epoch)):
        raise ValueError("invalid router spike epoch")
    after = _integer(raw.get("afterSequence"), "sequence")
    if after and epoch is None:
        raise ValueError("router spike continuation requires its capture epoch")
    transmission_epoch = raw.get("transmissionEpoch")
    if transmission_epoch is not None and (not isinstance(transmission_epoch, str)
            or not _EPOCH.fullmatch(transmission_epoch)):
        raise ValueError("invalid router transmission epoch")
    after_transmission = _integer(raw.get("afterTransmissionSequence"),
                                  "transmission sequence")
    if after_transmission and transmission_epoch is None:
        raise ValueError("router transmission continuation requires its selection epoch")
    transmission_page_size = _integer(
        raw.get("transmissionPageSize"), "transmission page size", 128,
        positive=True)
    if transmission_page_size > 128:
        raise ValueError("router transmission page size exceeds the exact transport envelope")
    return {"enabled": raw["enabled"], "capture_id": capture_id, "epoch": epoch,
            "start": _integer(raw.get("start"), "channel start"),
            "count": _integer(raw.get("count"), "channel count", 64, positive=True),
            "after": after, "page_size": min(_FRAME_PAGE, _integer(raw.get("pageSize"), "page size", _FRAME_PAGE, positive=True)),
            "transmission_epoch": transmission_epoch,
            "after_transmission": after_transmission,
            "transmission_page_size": transmission_page_size}


def _contact_query(raw):
    offset = _integer(raw.get("contactOffset"), "contact offset")
    page_size = _integer(raw.get("contactPageSize"), "contact page size", 128, positive=True)
    if page_size > 256:
        raise ValueError("router contact page size exceeds the exact transport envelope")
    return offset, page_size


def _empty(brain_id, capture_id, *, available=False, reason="", total=None,
           contact_offset=0, contact_page_size=128,
           transmission_page_size=128):
    return {"brainId": brain_id, "source": "actual-lif-hard-spikes", "owner": "root.router.population",
            "channelIdentity": "brainId:root.router.population:channelIndex", "available": available,
            "enabled": False, "observed": False, "captureId": capture_id, "epoch": None,
            "totalChannels": total, "channelStart": 0, "admittedCount": 0, "requestedCount": 0,
            "ringBytes": 0, "frameCapacity": 0, "lastSequence": 0, "firstAvailableSequence": 0,
            "nextSequence": 0, "droppedFrames": 0, "missedFrames": 0, "hasMore": False, "frames": [],
            "reason": reason, "semanticNeuronMapping": False, "synapseFiringObserved": False,
            "contactSnapshot": empty_router_contact_page(brain_id, reason=reason,
                offset=contact_offset, page_size=contact_page_size),
            "transmissionPage": empty_router_transmission_page(
                reason=reason, contact_offset=contact_offset,
                event_page_size=transmission_page_size)}


def _empty_for_query(brain_id, query, *, available=False, reason="", total=None,
                     contact_offset=0, contact_page_size=128):
    return _empty(brain_id, query["capture_id"], available=available,
                  reason=reason, total=total, contact_offset=contact_offset,
                  contact_page_size=contact_page_size,
                  transmission_page_size=query["transmission_page_size"])


def _snapshot_for_query(capture, query, contact_offset, contact_page_size):
    return capture.snapshot(
        epoch=query["epoch"], after=query["after"],
        page_size=query["page_size"], contact_offset=contact_offset,
        contact_page_size=contact_page_size,
        transmission_epoch=query["transmission_epoch"],
        after_transmission=query["after_transmission"],
        transmission_page_size=query["transmission_page_size"])


class RouterSpikeCapture:
    def __init__(self, brain_id, population, capture_id, start, count, reserve_ram,
                 contact_snapshot=None, *, synapses=None, ring_bytes=_RING_BYTES):
        if contact_snapshot is None:
            contact_snapshot = RouterContactSnapshot.unavailable(
                brain_id, start, count, "contact-snapshot-not-requested-by-direct-fixture")
        lease = None
        try:
            neurons = _integer(getattr(population, "neurons", None), "population size", positive=True)
            if start >= neurons:
                raise ValueError("router spike viewport exceeds the actual LIF population")
            admitted = min(count, neurons - start, _VIEWPORT_CHANNELS)
            frame_bytes = _HEADER.size + (admitted + 7) // 8
            capacity = max(1, ring_bytes // frame_bytes)
            allocated = capacity * frame_bytes
            # Escrow the bounded CPU conversion and outbound JSON page too. This
            # is an admission allowance, not a claim about exact process RSS.
            required = allocated + _METADATA_ALLOWANCE + admitted * 64 * (_FRAME_PAGE + 1)
            if not callable(reserve_ram):
                raise ValueError("router spike capture requires shared RAM admission")
            lease = reserve_ram(required, "transient LIF spike inspection ring")
            if lease is None or not callable(getattr(lease, "release", None)):
                raise ValueError("router spike capture requires a retained RAM lease")
            self._buffer = bytearray(allocated)
            self._lock = threading.RLock()
            self.population = weakref.ref(population)
            self.brain_id, self.capture_id = brain_id, capture_id
            self.start, self.count, self.requested_count, self.neurons = start, admitted, count, neurons
            self.frame_bytes, self.capacity = frame_bytes, capacity
            self.epoch = uuid.uuid4().hex
            self.sequence, self.closed, self.reason = 0, False, ""
            self._lease = lease
            self.contact_snapshot = contact_snapshot
            try:
                self.synapses = weakref.ref(synapses) if synapses is not None else lambda: None
            except TypeError:
                self.synapses = lambda: synapses
            if synapses is not None:
                contact_snapshot.bind_capture(self.epoch, synapses)
            # Keep this as retained escrow: outbound/slice scratch will be
            # allocated later. Claiming all of it already exists in RSS would
            # allow a subsequent sample to retire the promised scratch budget.
        except BaseException:
            if lease is not None and callable(getattr(lease, "release", None)):
                lease.release()
            contact_snapshot.close()
            raise

    def close(self, reason="capture-disabled"):
        with self._lock:
            if self.closed:
                return
            self.closed, self.reason = True, reason
            self._buffer = bytearray()
            lease, self._lease = self._lease, None
            contacts, self.contact_snapshot = self.contact_snapshot, None
            synapses = self.synapses() if hasattr(self, "synapses") else None
        if synapses is not None and getattr(synapses, "_native_transmission_observer", None) is self:
            synapses._native_transmission_observer = None
        if lease is not None:
            try:
                lease.release()
            except Exception as error:
                self.reason += "; inspection-lease-release-error: " + str(error)
        if contacts is not None:
            try:
                contacts.close()
            except Exception as error:
                self.reason += "; contact-inspection-lease-release-error: " + str(error)

    def record_hard_spikes(self, spikes):
        """Consume only the admitted slice of the actual pre-surrogate spikes."""
        with self._lock:
            if self.closed:
                return
            try:
                import torch
                if spikes.ndim != 1 or int(spikes.numel()) != self.neurons:
                    raise ValueError("actual LIF population geometry changed")
                selected = spikes.detach()[self.start:self.start + self.count].to(device="cpu")
                if not bool(((selected == 0) | (selected == 1)).all()):
                    raise ValueError("actual LIF spike payload is not binary")
                encoded = bytearray((self.count + 7) // 8)
                for index, value in enumerate(selected.to(dtype=torch.uint8).tolist()):
                    encoded[index // 8] |= value << (index % 8)
                if self.sequence >= _SAFE_INTEGER:
                    raise ValueError("router spike sequence exhausted its exact transport range")
                sequence = self.sequence + 1
                offset = (sequence - 1) % self.capacity * self.frame_bytes
                _HEADER.pack_into(self._buffer, offset, sequence, time.monotonic_ns(), time.time(), self.start, self.count)
                self._buffer[offset + _HEADER.size:offset + self.frame_bytes] = encoded
                self.sequence = sequence
            except Exception as error:
                # Inspection failure must not alter learned computation or
                # convert a successful neural step into a generation error.
                self.close("spike-capture-retired: " + str(error))

    def snapshot(self, *, epoch=None, after=0, page_size=_FRAME_PAGE,
                 contact_offset=0, contact_page_size=128,
                 transmission_epoch=None, after_transmission=0,
                 transmission_page_size=128):
        with self._lock:
            if epoch is not None and epoch != self.epoch:
                raise ValueError("router spike capture epoch changed; discard the old cursor")
            if after > self.sequence:
                raise ValueError("router spike cursor is ahead of this capture")
            base = _empty(self.brain_id, self.capture_id, available=True, reason=self.reason, total=self.neurons,
                          contact_offset=contact_offset, contact_page_size=contact_page_size,
                          transmission_page_size=transmission_page_size)
            first = max(1, self.sequence - self.capacity + 1) if self.sequence else 0
            start_sequence = max(after + 1, first)
            frames = []
            if not self.closed:
                for sequence in range(start_sequence, min(self.sequence + 1, start_sequence + page_size)):
                    offset = (sequence - 1) % self.capacity * self.frame_bytes
                    actual, monotonic, observed, channel, count = _HEADER.unpack_from(self._buffer, offset)
                    if actual != sequence or channel != self.start or count != self.count or not math.isfinite(observed):
                        raise ValueError("router spike ring identity diverged")
                    packed = self._buffer[offset + _HEADER.size:offset + self.frame_bytes]
                    if count % 8 and packed[-1] >> (count % 8):
                        raise ValueError("router spike ring payload is invalid")
                    values = [(packed[index // 8] >> (index % 8)) & 1 for index in range(count)]
                    frames.append({"sequence": sequence, "observedAtUnix": observed, "monotonicNs": str(monotonic),
                                   "channelStart": channel, "spikes": values})
            next_sequence = frames[-1]["sequence"] if frames else after
            contacts = (self.contact_snapshot.page(contact_offset, contact_page_size)
                if self.contact_snapshot is not None else empty_router_contact_page(
                    self.brain_id, reason="contact-snapshot-retired",
                    offset=contact_offset, page_size=contact_page_size))
            transmissions = (self.contact_snapshot.transmission_page(
                    contact_offset, contact_page_size,
                    epoch=transmission_epoch, after=after_transmission,
                    event_page_size=transmission_page_size)
                if self.contact_snapshot is not None else empty_router_transmission_page(
                    reason="router-transmission-snapshot-retired",
                    capture_epoch=self.epoch, contact_offset=contact_offset,
                    event_page_size=transmission_page_size))
            base.update(enabled=not self.closed, observed=self.sequence > 0, epoch=self.epoch,
                        channelStart=self.start, admittedCount=self.count, requestedCount=self.requested_count,
                        ringBytes=len(self._buffer), frameCapacity=self.capacity, lastSequence=self.sequence,
                        firstAvailableSequence=first, nextSequence=next_sequence,
                        droppedFrames=max(0, self.sequence - self.capacity), missedFrames=max(0, first - after - 1),
                        hasMore=not self.closed and next_sequence < self.sequence, frames=frames,
                        contactSnapshot=contacts, transmissionPage=transmissions)
            return base


def retire_population_spike_capture(population, reason):
    capture = getattr(population, "_native_spike_observer", None)
    if capture is not None:
        population._native_spike_observer = None
        capture.close(reason)


def retire_router_spike_capture(brain, reason="brain-closed", *, expected_capture=None):
    capture = getattr(brain, "_router_spike_capture", None)
    if expected_capture is not None and capture is not expected_capture:
        return
    if capture is not None:
        population = capture.population()
        if population is not None and getattr(population, "_native_spike_observer", None) is capture:
            population._native_spike_observer = None
        capture.close(reason)


def query_router_spikes(brain, brain_id: str, raw):
    query = _query(raw)
    contact_offset, contact_page_size = _contact_query(raw)
    if brain is None:
        return _empty_for_query(brain_id, query,
            reason="brain-not-loaded-no-inspection-model-created",
            contact_offset=contact_offset, contact_page_size=contact_page_size)
    if getattr(brain, "brain_id", None) != brain_id:
        raise ValueError("router spike capture brain identity differs")
    router = getattr(brain, "router", None)
    population = getattr(router, "population", None)
    synapses = getattr(router, "synapses", None)
    neurons = getattr(population, "neurons", None)
    if population is None or isinstance(neurons, bool) or not isinstance(neurons, int) or neurons < 1:
        return _empty_for_query(brain_id, query,
            reason="no-real-LIF-population-on-this-owner",
            contact_offset=contact_offset, contact_page_size=contact_page_size)
    capture = getattr(brain, "_router_spike_capture", None)
    same = capture is not None and capture.capture_id == query["capture_id"]
    if not query["enabled"]:
        if same:
            if query["epoch"] is not None and query["epoch"] != capture.epoch:
                raise ValueError("router spike cleanup epoch changed")
            retire_router_spike_capture(brain, "capture-disabled", expected_capture=capture)
            return _snapshot_for_query(capture, query, contact_offset, contact_page_size)
        return _empty_for_query(brain_id, query, available=True, total=neurons,
            reason="this-capture-is-not-active; other-captures-unchanged",
            contact_offset=contact_offset, contact_page_size=contact_page_size)
    if same and query["epoch"] is not None and query["epoch"] != capture.epoch:
        raise ValueError("router spike capture epoch changed; discard the old cursor")
    if same and capture.closed:
        return _snapshot_for_query(capture, query, contact_offset, contact_page_size)
    if same and (capture.population() is not population or capture.neurons != neurons
                 or capture.synapses() is not synapses or capture.start != query["start"]
                 or capture.requested_count != query["count"]):
        retire_router_spike_capture(brain, "population-or-viewport-changed; start-a-new-capture", expected_capture=capture)
        return _snapshot_for_query(capture, query, contact_offset, contact_page_size)
    if not same:
        # A continuation cannot silently replace a capture and replay its cursor
        # against a fresh population/session.
        if (query["epoch"] is not None or query["after"]
                or query["transmission_epoch"] is not None
                or query["after_transmission"]):
            raise ValueError("router spike capture owner changed; start a new capture")
        if query["start"] >= neurons:
            raise ValueError("router spike viewport exceeds the actual LIF population")
        policy = getattr(brain, "resource_policy", None)
        contact_snapshot = capture_router_contacts(
            brain, brain_id, query["start"],
            min(query["count"], neurons-query["start"], _VIEWPORT_CHANNELS))
        replacement = RouterSpikeCapture(brain_id, population, query["capture_id"], query["start"], query["count"],
                                         getattr(policy, "reserve_ram", None), contact_snapshot,
                                         synapses=synapses)
        retire_router_spike_capture(brain, "capture-replaced")
        brain._router_spike_capture = replacement
        population._native_spike_observer = replacement
        if contact_snapshot.available:
            synapses._native_transmission_observer = replacement
        capture = replacement
    return _snapshot_for_query(capture, query, contact_offset, contact_page_size)


def control_router_spikes(brain, brain_id: str, raw):
    """Metadata/bytes only: never allocates a capture or runs tensor code."""
    query = _query(raw)
    contact_offset, contact_page_size = _contact_query(raw)
    if brain is None:
        return _empty_for_query(brain_id, query,
            reason="brain-not-loaded-no-inspection-model-created",
            contact_offset=contact_offset, contact_page_size=contact_page_size)
    if getattr(brain, "brain_id", None) != brain_id:
        raise ValueError("router spike control brain identity differs")
    capture = getattr(brain, "_router_spike_capture", None)
    if capture is None or capture.capture_id != query["capture_id"]:
        return _empty_for_query(brain_id, query,
            reason="router-spike-capture-owner-changed; other-captures-unchanged",
            contact_offset=contact_offset, contact_page_size=contact_page_size)
    if query["epoch"] is None or query["epoch"] != capture.epoch:
        raise ValueError("router spike control requires the exact current capture epoch")
    router = getattr(brain, "router", None)
    population = getattr(router, "population", None)
    synapses = getattr(router, "synapses", None)
    if not capture.closed and (population is not capture.population()
            or synapses is not capture.synapses()
            or getattr(population, "neurons", None) != capture.neurons):
        retire_router_spike_capture(brain, "actual-LIF-owner-changed", expected_capture=capture)
    if not query["enabled"]:
        retire_router_spike_capture(brain, "capture-disabled", expected_capture=capture)
    return _snapshot_for_query(capture, query, contact_offset, contact_page_size)
