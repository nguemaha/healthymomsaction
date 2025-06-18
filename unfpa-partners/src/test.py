import requests

base = "https://www.ecfr.gov"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}
# Get structure of Title 21
structure_url = f"{base}/api/current/title-21"
resp = requests.get(structure_url)
data = resp.json()

# Navigate to Part 1 → Subpart B
subpart_b = None
for part in data["children"]:
    if part["label"].startswith("Part 1"):
        for subpart in part["children"]:
            if "Subpart B" in subpart["label"]:
                subpart_b = subpart
                break

# Get section identifiers in Subpart B
sections = subpart_b["children"]
print("Sections in Subpart B:")
for s in sections:
    identifier = s["identifier"]
    label = s["label"]
    print(f"{label} — {identifier}")

    # Fetch section content
    section_url = f"{base}/api/reader/v1/section/{identifier}"
    sec_resp = requests.get(section_url)
    sec_data = sec_resp.json()

    print("\n--- Section Text ---")
    print(sec_data["text"][:500], "...")  # Print first 500 chars
    print("--------------------\n")
