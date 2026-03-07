# 🧪 Inference Examples

Ready-to-use dialogue examples for testing the `/summarize` endpoint.

---

## Example 1 — Weekend Plans

**Input:**
```
Sam: Hey, are you free this weekend?
Alex: Saturday is packed but Sunday works.
Sam: Sunday afternoon? We could grab brunch.
Alex: Sounds perfect, let's say 11am at Maple Cafe.
Sam: Great, see you then!
```

**Baseline Summary:**
> Alex and Sam will meet at Maple Cafe on Sunday at 11 am.

**curl:**
```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Sam: Hey, are you free this weekend?\nAlex: Saturday is packed but Sunday works.\nSam: Sunday afternoon? We could grab brunch.\nAlex: Sounds perfect, lets say 11am at Maple Cafe.\nSam: Great, see you then!", "max_length": 128}'
```

---

## Example 2 — Meeting Reschedule

**Input:**
```
Manager: The client call is moved to Thursday 3pm.
Dev: I have a conflict at 3, can we do 4pm instead?
Manager: Let me check... yes, 4pm works.
Dev: Perfect. Should I prepare a demo?
Manager: Yes, focus on the new dashboard features.
Dev: Got it, I'll have slides and a live demo ready.
```

**Baseline Summary:**
> Dev will prepare a demo at 4 pm on Thursday.

**curl:**
```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Manager: The client call is moved to Thursday 3pm.\nDev: I have a conflict at 3, can we do 4pm instead?\nManager: Let me check... yes, 4pm works.\nDev: Perfect. Should I prepare a demo?\nManager: Yes, focus on the new dashboard features.\nDev: Got it, Ill have slides and a live demo ready.", "max_length": 128}'
```

---

## Example 3 — Grocery Delivery

**Input:**
```
Mia: Did the grocery delivery arrive?
Tom: Yes, but they substituted the almond milk with oat milk.
Mia: That's fine, I actually prefer oat milk.
Tom: Also they were out of sourdough, sent whole wheat instead.
Mia: Ugh, I really wanted sourdough for the weekend.
Tom: I can swing by the bakery on my way home.
Mia: That would be amazing, thanks!
```

**Baseline Summary:**
> Tom will pick up the groceries on Mia's request.

**curl:**
```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Mia: Did the grocery delivery arrive?\nTom: Yes, but they substituted the almond milk with oat milk.\nMia: Thats fine, I actually prefer oat milk.\nTom: Also they were out of sourdough, sent whole wheat instead.\nMia: Ugh, I really wanted sourdough for the weekend.\nTom: I can swing by the bakery on my way home.\nMia: That would be amazing, thanks!", "max_length": 128}'
```
