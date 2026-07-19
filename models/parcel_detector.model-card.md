# Parcel Detector Baseline Model Card

- Artifact: `parcel_detector.onnx`
- SHA-256: `f695550212d2dd2059928f543f3550690527bc0ed7071dca040e5f12c849e432`
- Architecture: Ultralytics YOLOv8n, fixed 640 x 640 input
- Upstream dataset/model: [Roboflow Logistics](https://universe.roboflow.com/large-benchmark-datasets/logistics-sz9jr), version 2
- Checkpoint mirror: [MTerryJack/Element-Detect-WarehouseLogistics](https://huggingface.co/MTerryJack/Element-Detect-WarehouseLogistics)
- Dataset license: CC BY 4.0
- Embedded model metadata license: AGPL-3.0; review Ultralytics licensing obligations before commercial deployment
- Upstream reported metrics: mAP@50 76.6%, precision 79.5%, recall 70.7% across 20 logistics classes

## Runtime mapping

This is a conservative baseline, not the repository's planned five-class custom detector.
The runtime maps `cardboard box` to parcel evidence, uses image-edge contact to mark
partial visibility, maps `barcode`/`qr code` to candidate crop regions, and ignores
unrelated classes. It does not claim to detect blocked labels. Existing OCR, barcode
decoding, label heuristics, reconciliation, and review gates remain active.

## Validation status

The artifact has been checked for ONNX Runtime CPU loading and the application's
expected YOLO tensor layout. It has not been accuracy-validated on the private target
warehouse dataset. A custom model trained on the five repository classes should replace
this baseline after annotation, frozen evaluation, and acceptance review.
