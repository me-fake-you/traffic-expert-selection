# Data and privacy boundaries

No raw dataset, PCAP, endpoint, payload, model checkpoint or row-level label/prediction file is included. Aggregate confusion counts and finite-candidate summaries are included to make the reported comparisons inspectable. Dataset redistribution rights are not asserted by this code package.

## Populations must stay separate

| Population | Unit and size | Permitted interpretation |
|---|---|---|
| USTC-TFC2016 controlled comparison | 40,000 sampled segments; 20 application/family groups | Grouped, controlled expert-selection replay on an exposed benchmark |
| Same-source raw-input execution | 625,523 common eligible segments from 24 captures | Natural-frequency execution and one-run processing costs, not an improved model |
| Intersection with the original 40k | 39,428 eligible; 572 unsupported in the raw path | The 572 comprise 569 ARP, one ARP/Ethernet and two ICMPv6 segments; old generic features do not establish new-path support |
| IoT-23 frozen transfer | 1,159,418 label-matched author flows, 465,990 benign; seven dependent captures | External source and sample-unit shift; high false-alarm failure retained |

USTC labels are inherited from captures/application-family assignments; they are not packet-level certification of an attack. The IoT-23 evaluation follows author flow labels and its explicit matching contract. Refer to the manuscript bibliography and the original data providers for the exact releases and terms. Obtain data from providers under their terms; this repository does not grant data rights or substitute predicted labels for ground truth.

The complete system architecture is contextual. TLS processing, advisers, Yager fusion and four-way outputs are not evaluated by the binary expert-selection comparison. Strategy Calls is a count, not an end-to-end runtime measurement.
