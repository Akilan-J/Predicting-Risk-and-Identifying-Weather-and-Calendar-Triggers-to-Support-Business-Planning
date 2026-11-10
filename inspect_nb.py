import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open(r'C:\Users\jxiak\Documents\7th Sem\business analytics\group study\m4\M4_Feature_Engineering.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']

# Print key cells
for idx in [2, 17, 19, 21, 25, 27, 29, 31, 33, 42]:
    print(f"\n{'='*80}")
    print(f"CELL {idx} [{cells[idx]['cell_type']}]:")
    src = ''.join(cells[idx]['source'])
    print(src[:2000])
