Rubric: **binary**, judge: `qwen2.5:7b`, n=300

| arm | stud→stud | polite→polite | trait accuracy | separation |
|---|---|---|---|---|
| `vanilla_prompt` | 35% | 70% | 52% | 5% |
| `vanilla_bare` | - | - | - | - |
| `head_shallow` | 100% | 100% | 100% | 100% |
| `head_deep` | 100% | 100% | 100% | 100% |
| `lora` | 100% | 95% | 98% | 95% |
| `vanilla_prompt_long` | 20% | 85% | 52% | 5% |
| `vanilla_prompt_plain` | 80% | 40% | 60% | 20% |

| arm | requested | judged stud | mean words | repetition | looped |
|---|---|---|---|---|---|
| `vanilla_prompt` | stud | 35% | 68 | 0.00 | 0% |
| `vanilla_prompt` | polite | 30% | 67 | 0.00 | 0% |
| `vanilla_bare` | none | 45% | 32 | 0.00 | 0% |
| `head_shallow` | stud | 100% | 22 | 0.00 | 0% |
| `head_shallow` | polite | 0% | 25 | 0.00 | 0% |
| `head_shallow` | bare | 95% | 28 | 0.00 | 0% |
| `head_deep` | stud | 100% | 21 | 0.00 | 0% |
| `head_deep` | polite | 0% | 28 | 0.00 | 0% |
| `head_deep` | bare | 95% | 22 | 0.00 | 0% |
| `lora` | stud | 100% | 22 | 0.00 | 0% |
| `lora` | polite | 5% | 26 | 0.00 | 0% |
| `vanilla_prompt_long` | stud | 20% | 299 | 0.00 | 0% |
| `vanilla_prompt_long` | polite | 15% | 224 | 0.00 | 0% |
| `vanilla_prompt_plain` | stud | 80% | 38 | 0.00 | 0% |
| `vanilla_prompt_plain` | polite | 60% | 56 | 0.00 | 0% |
