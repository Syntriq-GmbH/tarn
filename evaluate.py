import argparse
import json
import re

import torch
from datasets import load_dataset

LABELS = ["ADDRESS", "CREDIT_CARD", "CUSTOMER_ID", "DATE_OF_BIRTH", "DE_BSNR", "DE_FUEHRERSCHEIN", "DE_HEALTH_INSURANCE", "DE_ID_CARD", "DE_KFZ", "DE_LANR", "DE_PASSPORT", "DE_SOCIAL_SECURITY", "DE_TAX_ID", "DE_TAX_NUMBER", "EMAIL_ADDRESS", "IBAN_CODE", "IP_ADDRESS", "PERSON", "PHONE_NUMBER"]
GLINER = {
    "person": "PERSON", "email address": "EMAIL_ADDRESS", "phone number": "PHONE_NUMBER", "address": "ADDRESS", "date of birth": "DATE_OF_BIRTH",
    "iban": "IBAN_CODE", "credit card number": "CREDIT_CARD", "ip address": "IP_ADDRESS", "customer number": "CUSTOMER_ID",
    "tax identification number": "DE_TAX_ID", "tax number": "DE_TAX_NUMBER", "social security number": "DE_SOCIAL_SECURITY",
    "health insurance number": "DE_HEALTH_INSURANCE", "identity card number": "DE_ID_CARD", "passport number": "DE_PASSPORT",
    "driver's license number": "DE_FUEHRERSCHEIN", "license plate number": "DE_KFZ", "doctor number": "DE_LANR", "medical practice number": "DE_BSNR",
}
MAPPINGS = {
    "bardsai/eu-pii-anonimization-multilang": {
        "PERSON_NAME": "PERSON", "EMAIL_ADDRESS": "EMAIL_ADDRESS", "PHONE_NUMBER": "PHONE_NUMBER", "POSTAL_ADDRESS": "ADDRESS", "LOCATION": "ADDRESS",
        "DATE_OF_BIRTH": "DATE_OF_BIRTH", "BANK_ACCOUNT_IDENTIFIER": "IBAN_CODE", "PAYMENT_CARD": "CREDIT_CARD", "IP_ADDRESS": "IP_ADDRESS",
        "VEHICLE_IDENTIFIER": "DE_KFZ", "ACCOUNT_IDENTIFIER": "CUSTOMER_ID", "DOCUMENT_IDENTIFIER": "PII", "PERSON_IDENTIFIER": "PII", "CONTACT_HANDLE": "PII",
    },
    "tabularisai/eu-pii-safeguard": {
        "FIRSTNAME": "PERSON", "MIDDLENAME": "PERSON", "LASTNAME": "PERSON", "EMAIL": "EMAIL_ADDRESS", "PHONE_NUMBER": "PHONE_NUMBER",
        "ADDRESS": "ADDRESS", "STREET": "ADDRESS", "DOB": "DATE_OF_BIRTH", "IBAN": "IBAN_CODE",
        "CREDIT_CARD_NUMBER": "CREDIT_CARD", "IP_ADDRESS": "IP_ADDRESS", "TAX_ID": "DE_TAX_ID", "HEALTH_INSURANCE_ID": "DE_HEALTH_INSURANCE",
        "PASSPORT_NUMBER": "DE_PASSPORT", "DRIVER_LICENSE": "DE_FUEHRERSCHEIN", "NATIONAL_ID": "PII", "ACCOUNT_NUMBER": "PII",
    },
    "openai/privacy-filter": {
        "private_person": "PERSON", "private_email": "EMAIL_ADDRESS", "private_phone": "PHONE_NUMBER", "private_address": "ADDRESS",
        "private_date": "DATE_OF_BIRTH", "account_number": "PII",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="syntriq/tarn-german-pii-base")
parser.add_argument("--gliner", action="store_true")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--limit", type=int)
parser.add_argument("--output")
args = parser.parse_args()

documents = load_dataset("syntriq/german-pii-benchmark", split="test")
documents = documents.select(range(args.limit)) if args.limit else documents
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
predicted = []

if args.gliner:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(args.model).to(device)
    for text in documents["text"]:
        spans, start = [], 0
        while True:
            end = start + 1000
            spans += [
                {"start": start + entity["start"], "end": start + entity["end"], "label": GLINER[entity["label"]]}
                for entity in model.predict_entities(text[start:end], list(GLINER), threshold=args.threshold)
                if (start == 0 or entity["start"] >= 150) and (end >= len(text) or entity["start"] < 850)
            ]
            if end >= len(text):
                break
            start += 700
        predicted.append(spans)
else:
    from transformers import pipeline

    mapping = MAPPINGS.get(args.model, {label: label for label in LABELS})
    detector = pipeline("token-classification", model=args.model, aggregation_strategy="simple", stride=128 if args.model.startswith("syntriq/") else 64, device=device)
    for text in documents["text"]:
        predicted.append([{"start": entity["start"], "end": entity["end"], "label": mapping[entity["entity_group"]]} for entity in detector(text) if entity["entity_group"] in mapping])

cleaned = []
for text, spans in zip(documents["text"], predicted):
    merged = []
    for span in sorted(spans, key=lambda span: span["start"]):
        value = text[span["start"]:span["end"]]
        span = {**span, "start": span["start"] + len(value) - len(value.lstrip()), "end": span["end"] - len(value) + len(value.rstrip())}
        gap = text[merged[-1]["end"]:span["start"]] if merged else None
        if merged and span["label"] == merged[-1]["label"] and gap is not None and re.fullmatch(r"[ \t]*" if span["label"] != "ADDRESS" else r"[\s,]*", gap):
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        elif span["start"] < span["end"] and not (merged and span["start"] < merged[-1]["end"]):
            merged.append(span)
    cleaned.append(merged)

counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
masked = {"gold": 0, "gold_hit": 0, "predicted": 0, "predicted_hit": 0}
for gold_entities, predicted_entities in zip(documents["entities"], cleaned):
    expected = {(entity["start"], entity["end"], entity["label"]) for entity in gold_entities}
    found = {(entity["start"], entity["end"], entity["label"]) for entity in predicted_entities}
    for span in found:
        if span[2] in counts:
            counts[span[2]]["tp" if span in expected else "fp"] += 1
    for span in expected - found:
        counts[span[2]]["fn"] += 1
    masked["gold"] += len(expected)
    masked["gold_hit"] += sum(any(start < other_end and other_start < end for other_start, other_end, _ in found) for start, end, _ in expected)
    masked["predicted"] += len(found)
    masked["predicted_hit"] += sum(any(start < other_end and other_start < end for other_start, other_end, _ in expected) for start, end, _ in found)

f1 = lambda c: 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1)
total = {key: sum(c[key] for c in counts.values()) for key in ("tp", "fp", "fn")}
result = {
    "model": args.model,
    "documents": len(documents),
    "masked_recall": masked["gold_hit"] / max(masked["gold"], 1),
    "masked_precision": masked["predicted_hit"] / max(masked["predicted"], 1),
    "strict_f1": f1(total),
    "labels": {label: {"f1": f1(c), "support": c["tp"] + c["fn"]} for label, c in counts.items()},
}
print(f"{args.model}: masked recall {result['masked_recall']:.3f}, masked precision {result['masked_precision']:.3f}, strict F1 {result['strict_f1']:.3f}")
if args.output:
    json.dump(result, open(args.output, "w"), indent=2)
