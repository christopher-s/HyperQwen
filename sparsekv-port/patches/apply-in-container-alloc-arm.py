p = '/app/single-user/start_qwen.sh'
s = open(p).read()
old = '''case " ${EXTRA_ARGS:-} " in
    *"__never_match_sparsekv_port"*)
    [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && [ "$ALLOC_DEFAULT" = expandable_segments:True ] && echo "KV connector in EXTRA_ARGS: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False (vLLM rejects the connector under VMM; set it explicitly to override)"
    ALLOC_DEFAULT=expandable_segments:False ;;
esac'''
new = '''case " ${EXTRA_ARGS:-} " in
    *" --kv-transfer-config"*)
    [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && [ "$ALLOC_DEFAULT" = expandable_segments:True ] && echo "KV connector in EXTRA_ARGS: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False (vLLM rejects the connector under VMM; set it explicitly to override)"
    ALLOC_DEFAULT=expandable_segments:False ;;
esac'''
if old in s:
    open(p, 'w').write(s.replace(old, new, 1))
    print('allocator arm restored')
else:
    print('anchor drifted')
