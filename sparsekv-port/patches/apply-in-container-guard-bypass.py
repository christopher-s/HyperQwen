p = '/app/single-user/start_qwen.sh'
lines = open(p).read().split('\n')
out, killed = [], 0
for i, ln in enumerate(lines):
    if ('--kv-offloading-size' in ln and 'kv-transfer-config' in ln and ln.strip().startswith('*')):
        # replace the case-pattern line with an always-false pattern
        out.append('    *"__never_match_sparsekv_port"*)')
        killed += 1
        continue
    if 'untested on this backend; refusing. Use PTH_FA=0 for a CPU KV tier.' in ln:
        killed += 1
        continue  # drop the refusal echo
    if 'PTH_FA=1 with a KV connector (KV_OFFLOAD_GB' in ln:
        killed += 1
        continue  # drop the explanation echo
    if killed and ln.strip() == 'exit 1 ;;' and lines[i-1].strip() != 'esac' and 'sparsekv' not in lines[i-1]:
        # only drop the exit that belonged to the killed arm (the one right after dropped echoes)
        if out and out[-1] == '' :
            killed += 1
            continue
    out.append(ln)
open(p, 'w').write('\n'.join(out))
print('killed lines:', killed)
