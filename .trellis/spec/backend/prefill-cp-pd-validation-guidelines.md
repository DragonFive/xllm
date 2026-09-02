# Prefill Context Parallel and PD Validation Contract

## 1. Scope / Trigger

Apply this contract when a text model uses Prefill context parallelism (PCP),
owner-sharded KV cache, disaggregated Prefill/Decode (PD), or a composed MTP or
graph path. It covers Python/TORCH PCP, native model-side CP, KV-split routing,
Mooncake or LlmDataDist transfer, Decode readiness, numerical comparison, and
runtime evidence.

This contract is required whenever `cp_size > 1`, `kv_split_size > 1`, a PD
transfer backend is enabled, or a change claims that PCP composes with MTP,
graph execution, Prefix Cache, chunked Prefill, or concurrency.

## 2. Signatures

The supported Python PCP Prefill shape is explicit and eager:

```text
--model_impl=python
--cp_size=<N greater than 1>
--dp_size=1
--enable_graph=false
--python_graph_backend=off
```

For GLM owner-sharded KV, `kv_split_size` must be a positive divisor of
`cp_size`. A `kv_split_size > 1` deployment requires the PD Prefill role; it is
not a standalone Decode or default-role shortcut.

MTP after Python PCP belongs on the Decode side:

```text
Prefill: target model, cp_size>1, MTP off, graph off
Decode:  target + draft model, cp_size=1,
         --speculative_algorithm=MTP,
         --num_speculative_tokens=<positive value>
```

A backend A/B must set the same value on both roles:

```text
Prefill --kv_cache_transfer_type=<Mooncake|LlmDataDist>
Decode  --kv_cache_transfer_type=<the same value>
```

Binary Mooncake Decode notifications use an ASCII-safe, versioned envelope:

```text
xllm.decode-kv-notification.v1:<raw_size>:<base64_payload>
```

An uncommitted runtime claim must identify the executable source with at least:

```text
HEAD <commit>
TRACKED_DIFF_SHA256 <sha256>
UNTRACKED_SHA256 <sha256> <path>   # one sorted line per source file
SOURCE_MANIFEST_SHA256 <sha256 of the newline-terminated manifest>
```

The wheel archive hash, the extracted `xllm/xllm` payload hash, and the live
executable hash are different identities and must be recorded separately.

## 3. Contracts

### Compute, storage, and transfer planes

Keep these planes distinct:

1. Query/token CP determines which query rows a rank computes.
2. Current-forward K/V may be temporarily gathered for one attention call.
3. Persistent KV ownership determines which rank writes each physical slot.
4. PD transfer routes each persistent owner shard to the matching Decode
   physical destination.

Temporary materialization does not replicate persistent ownership. Decode does
not need a second reconstruction kernel when Prefill owners write the complete
Decode physical layout, but that conclusion requires matching owner, slot,
block-table, layer, role, and partial-tail semantics.

For each request, verify contributions over the full identity:

```text
request x layer x owner shard x cache role x destination
```

`KEY`, `VALUE`, and `INDEX` are independent roles. Source submission for one
role or one layer is not evidence of complete handoff.

### Failure localization and backend isolation

Classify the first failed boundary before changing mapping code:

```text
startup -> registration -> transport preparation -> PUSH submission
        -> transport completion -> receiver receipt -> Decode enqueue
        -> model execution -> numerical comparison
```

For example, an `HcclCommPrepare` failure before the first PUSH completes is
transport/fabric evidence. It does not prove that owner-slot or `kv_split`
mapping is wrong.

Use a strict, two-sided backend A/B with the same source, wheel, model,
request, topology, ports, and optional features. Interpret it as follows:

- Mooncake fails and LlmDataDist passes: investigate backend/fabric
  compatibility.
- Both fail: investigate host fabric, ranktable, ports, registration, and
  schema before changing CP mapping.
- Both transfer but Decode output differs: investigate owner slots, physical
  blocks, cache roles, and partial-tail validity.

### Notification and readiness

Never place arbitrary protobuf bytes directly in a JSON string. Encode binary
payloads with a versioned ASCII-safe frame, validate the declared raw size, and
fail the entire drained payload on a malformed frame. A compatibility receiver
may retain legacy passthrough, but rolling deployment must coordinate sender
and receiver versions.

Decode enqueue requires receiver-visible, request-scoped evidence when the
backend provides it. Source-side API success or PUSH completion alone is not a
strict readiness proof.

Missing, malformed, stale, or conflicting receipts fail closed. If remote DMA
may still target the allocation and the backend lacks a request-scoped
cancellation/quiescence fence, quarantine the request and its KV allocation;
do not recycle the blocks after a time-only grace period.

### Prefix and partial-tail units

Name every block-count unit. Decode destination blocks and Prefill source
blocks may differ by `kv_split_size`; convert before applying Prefix Cache
coverage. A partial final block contributes only valid logical token rows and
must not create a remote mapping for an absent destination.

### PCP, MTP, and graph composition

Python/TORCH PCP is eager-Prefill-only. With `cp_size > 1`:

- reject MTP speculative verification in the same instance;
- reject ACLGraph and arbitrary `torch.compile` graph backends;
- keep MTP on a `cp_size=1` Decode instance after PD handoff;
- do not treat native ATB/TORCH model-side CP capability as proof that the
  Python executor supports the same combination.

Native model-side CP may use a different phase-disjoint contract in which CP
acts only during Prefill and ACLGraph acts only during pure Decode. That is a
separate implementation and requires its own model-specific tests.

### Numerical and experimental causality

Compare equal logical views, not raw rank-local storage. Reconstruct
owner-sharded physical KV into global logical token order, remove padding and
invalid tail rows, and compare hidden states, logits, Indexer outputs, and all
persistent cache roles as applicable.

A causal A/B changes one intended variable. If CP, TP, world size, model,
backend, or request changes together, label the result observational. Compute
topology and checkpoint-memory feasibility before launching; an expected OOM
does not add correctness evidence.

### Artifact and evidence durability

Before traffic, bind the source manifest to a fresh wheel, extracted payload,
installed executable, and live rank command line. A package version or archive
name is insufficient.

Preserve small completion-critical artifacts under the owning task with a
path-sorted `SHA256SUMS`: request/response, bounded request-window logs,
coverage, comparator, identity, health, and cleanup reports. Treat `/dev/shm`,
container-local paths, and remote-only narratives as volatile.

HTTP 200 proves request lifecycle only. It does not by itself prove logical KV
correctness, receiver readiness, semantic quality, performance, or graceful
shutdown.

## 4. Validation & Error Matrix

| Condition | Required behavior | Forbidden conclusion |
| --- | --- | --- |
| `HcclCommPrepare` fails before transfer completion | Classify transport/fabric and preserve the first failure | Do not claim a `kv_split` mapping bug |
| Only one PD role changes backend | Reject the experiment as invalid | Do not compare backend behavior |
| Binary notification is carried as a raw JSON string | Replace it with a versioned binary-safe frame | Do not only increase a size limit |
| Source submission succeeds without receiver evidence | Record source coverage only | Do not claim strict Decode readiness |
| Required receipt is missing or conflicting | Timeout/poison, no Decode enqueue, quarantine if DMA may remain live | Do not partially publish |
| Prefix source/destination block units differ | Convert through the documented `kv_split_size` relation | Do not compare raw counts |
| Python PCP requests MTP verification | Reject before model execution and direct MTP to CP1 Decode | Do not construct the draft and fail mid-request |
| Python PCP requests graph mode | Reject before model execution and require eager Prefill | Do not silently run without CP sharding |
| CP and TP both change in a numerical A/B | Report a confounded observation | Do not attribute the result to CP alone |
| Full-model baseline cannot fit model weights | Record the capacity proof and use layered causal evidence | Do not occupy devices for an expected OOM |
| Wheel payload differs from the live executable | Stop testing and deploy the intended artifact | Do not attribute output to current source |
| Errors occur only after traffic during forced cleanup | Keep them outside the request-window verdict | Do not claim graceful shutdown either |

## 5. Good / Base / Bad Cases

- Good: run eager Python PCP on Prefill, transfer every owner/layer/role,
  require receiver-visible completion, then run Decode or MTP at `cp_size=1`.
- Good: hold every variable fixed while switching both PD roles from Mooncake
  to LlmDataDist, and report source versus receiver evidence separately.
- Good: compare CP/KV variants after reconstructing the same logical token
  order, with identical model, request, TP, backend, and executable.
- Base: run a same-host compact lifecycle smoke before cross-host transport;
  label HTTP 200 as lifecycle evidence only.
- Bad: infer mapping failure from an HCCL preparation error, change CP code,
  and rerun with a different host, model, and backend.
- Bad: enable Python PCP, MTP verification, and graph together and rely on
  eager fallback. Unsupported combinations must fail early.
- Bad: preserve the only receipt or comparator in `/dev/shm`, clean the
  service, and retain only a narrative PASS statement.

## 6. Tests Required

1. Topology tests cover `cp_size`, every valid divisor `kv_split_size`, owner
   rank/local slot mapping, destination stride, and partial final blocks.
2. Batch/metadata round trips preserve request, layer, role, owner, block, and
   destination identity without duplicate or missing rows.
3. Notification tests round-trip arbitrary protobuf bytes through the
   versioned frame and reject bad version, size, Base64, and payload data.
4. Readiness tests cover delayed final receipt, missing/stale/conflicting
   receipt, malformed notification, worker failure, late registration,
   exactly-once publish, quarantine, and subsequent distinct-request progress.
5. A compact dual-instance run proves per-layer/per-role/per-owner transfer and
   a partial tail before the full-model gate.
6. A two-sided Mooncake/LlmDataDist A/B holds all non-backend inputs fixed and
   states whether evidence is source-submission or receiver-visible.
7. Numerical tests reconstruct logical KV and compare hidden/logits/Indexer at
   fixed model, request, TP, and backend. Confounded comparisons are non-gating.
8. Prefix Cache, chunked Prefill, same-prefix concurrency, and packed-prefix
   concurrency are separate composition rows; compact evidence is not promoted
   to full-model evidence.
9. A real-backend negative withholds one exact receipt and proves success,
   fail-closed/no-enqueue, and a later success without service restart. Remove
   every test-only hook before producing the final source manifest.
10. Admission tests independently assert Python PCP-only success, PCP+MTP
    rejection, PCP+Graph rejection, and CP1 behavior.
11. Build and runtime tests record source manifest, wheel archive, payload,
    installed/live executable, model, topology, ports, flags, and bounded log
    windows.
12. A final task-local audit parses structured artifacts, verifies every
    checksum, and lists untested semantic, performance, and composition scopes.

## 7. Wrong vs Correct

Wrong failure attribution:

```text
HcclCommPrepare failed -> change kv_split owner mapping
```

Correct failure attribution:

```text
HcclCommPrepare failed before first PUSH completion
-> validate fabric/address planes/registration
-> run a two-sided backend A/B
-> inspect owner mapping only after transfer succeeds but logical output fails
```

Wrong feature composition:

```text
Python Prefill: cp_size=4, speculative_algorithm=MTP, enable_graph=true
```

Correct role separation:

```text
Prefill: model_impl=python, cp_size=4, MTP off, graph off
Decode:  cp_size=1, target + draft, speculative_algorithm=MTP
```

Wrong numerical comparison:

```text
Compare CP1 rank-local KV directly with one CP4 owner shard.
```

Correct numerical comparison:

```text
Reconstruct CP4 owner-local physical slots into global logical token order,
drop padding/invalid tail rows, then compare the same roles and token positions.
```

Wrong deployment identity:

```text
wheel filename and package version match, so the live ranks use current code
```

Correct deployment identity:

```text
source manifest -> fresh wheel archive -> extracted payload
                -> installed executable -> every live rank command/path/hash
```
