# Effect Safety Template

| Failure point | Method | Final-state correct ↑ | Exactly-once effects ↑ | Duplicate effects ↓ | Conflicting effects ↓ | Recovery latency ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Before send | No ledger | TBD | TBD | TBD | TBD | TBD |
| Before send | Dedupe-only | TBD | TBD | TBD | TBD | TBD |
| Before send | Commit barrier | TBD | TBD | TBD | TBD | TBD |
| Before send | AgentRelay | TBD | TBD | TBD | TBD | TBD |
| After send / before ack | AgentRelay | TBD | TBD | TBD | TBD | TBD |
| After ack / before packet commit | AgentRelay | TBD | TBD | TBD | TBD | TBD |
| During target retry | AgentRelay | TBD | TBD | TBD | TBD | TBD |

