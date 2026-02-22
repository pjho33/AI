# ColabFold Settings for Q70JN9 sldA

---

## ✅ Sequence Input (Correct)

Screenshot shows sequence is already entered in `query_sequence` field.

**Verify it matches**:
```
BKKADIELGLMPIDDPNYKPGGGGAEGNGAMDGTPYGIVVTPFWDQYTGMMCNRPPYGMITAIDMKHGQKVL
WQHPLGTARANGPWGLPTGLPWEIGTPNNGGSVVTAGGVVFIAAATDNQIRAIDEHTGKVVWSAVLPGGGQA
NPMTYEANGHQYVAIMAGGHHFMMTPVSDQLVVYALPDHKG
```

**Note**: This appears to be only the C-terminal portion. Need to check if full sequence was entered.

---

## 📋 Settings to Configure

### 1. jobname
**Current**: `test`  
**Recommended**: `Q70JN9_sldA_catalytic`

### 2. num_relax
**Current**: `0`  
**Recommended**: `5`
- Relaxes top 5 models with AMBER
- Improves geometry

### 3. template_mode
**Current**: `none`  
**Keep**: `none`
- No template needed
- Pure AlphaFold prediction

---

## ⚠️ Important Check

**The sequence shown in screenshot appears incomplete!**

Expected length: **713 aa**  
Visible in screenshot: Only ~200 aa (C-terminal portion)

### Action Required

**Scroll up in the query_sequence box** to verify the full sequence is there:

**Full sequence should start with**:
```
>Q70JN9_sldA_catalytic_domain|residues_31-end
SGGSPTSSVPGPGNGSGNSFEPTENTPAAKSRFSGPSPYAPQAPGVNAANLPDIGSMDPN
DVPQMAPQQSASPASGDWAAYGHDDSQMRY...
```

**And end with**:
```
...NPMTYEANGHQYVAIMAGGHHFMMTPVSDQLVVYALPDHKG
```

---

## ✅ Recommended Settings

```
query_sequence: [Full 713 aa sequence from sequence.fasta]
jobname: Q70JN9_sldA_catalytic
num_relax: 5
template_mode: none
```

---

## 🚀 After Settings

1. **Verify full sequence** (scroll to check)
2. **Update settings** as above
3. **Runtime → Run all**
4. **Wait 10-30 min**
5. **Download best model** (`*_rank_1_*.pdb`)
6. **Save as** `structures/receptor_af.pdb`

---

## Expected Output

- 5 models generated
- Ranked by confidence
- Best model: `*_rank_1_*.pdb`
- pLDDT scores in B-factor column
- Should show β-propeller fold

---

## Next Step After AlphaFold

Once you have `receptor_af.pdb`:
→ **PQQ+Ca transplant** using ChimeraX (see `PQQ_TRANSPLANT_GUIDE.md`)
