"""One-shot MTP graft. Offsets are data-section-relative and INVARIANT under header rewrite:
copy the data section verbatim, keep base offsets, append blobs, compute their rel offsets."""
import json, struct, os, glob, sys

target = sys.argv[1]
orig_dir = sys.argv[2]

def read_header(p):
    with open(p,'rb') as f:
        m = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(m)), 8+m

hdr, ds = read_header(target)
base_len = os.path.getsize(target) - ds

blobs = {}
for f in sorted(glob.glob(orig_dir+'/*.safetensors')):
    h, dstart = read_header(f)
    for k in h:
        if k.startswith('mtp.'):
            a,b = h[k]['data_offsets']
            with open(f,'rb') as sf:
                sf.seek(dstart+a)
                blobs[k] = (h[k], sf.read(b-a))
assert len(blobs)==15, len(blobs)

new_hdr = {k: dict(v) for k,v in hdr.items()}
rel = base_len
for k in sorted(blobs):
    meta, blob = blobs[k]
    new_hdr[k] = dict(meta, data_offsets=[rel, rel+len(blob)])
    rel += len(blob)

hjson = json.dumps(new_hdr, separators=(',',':')).encode()
tmp = target+'.new'
with open(target,'rb') as src, open(tmp,'wb') as out:
    out.write(struct.pack('<Q', len(hjson)))
    out.write(hjson)
    src.seek(ds)
    remaining = base_len
    while remaining:
        c = src.read(min(1<<24, remaining))
        if not c: break
        out.write(c); remaining -= len(c)
    for k in sorted(blobs):
        out.write(blobs[k][1])
os.replace(tmp, target)
print(f'grafted 15 mtp tensors. base {base_len}B + grafts, header {len(hjson)}B, file {os.path.getsize(target)}B')
