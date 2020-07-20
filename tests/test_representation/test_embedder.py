from mlmc.representation import Embedder

test_models = ("bert", "roberta",)
example = ["This sentence", "another !!!!! potsasldkaölskd ölaskdölaskdz!"]

def test_embedder():
    for mod in test_models:
        e = Embedder(mod)
        results = {}
        for m in e.methods:
            e.set_method(m)
            results[m] = e.embed(example)

        assert [x.shape[0] for x in results["pool"]] == [x.shape[0] for x in results["first_token"]] == [len(x.split()) for x in example], \
            f"Pool and first_token result in different shapes for {mod}"
        assert all(len(v)==len(results["pool"]) for v in results.values()), \
            f"Not all methods return the same number of sentences for {mod}"
        assert all(v[0].shape[-1]==results["pool"][0].shape[-1] for v in results.values()), \
            f"Not all methods return the same embedding dimension for {mod}"

        results = {}
        for m in e.methods:
            e.set_method(m)
            results[m] = e.embed(example, pad = 100)


        assert all(v.shape[0]==len(example) for v in results.values())
        assert all(v.shape[-1]==results["sentence"].shape[-1] for v in results.values())
        assert all(v.shape[1]==results["pool"].shape[1] for v in results.values() if len(v.shape)==3), f"Padding error for {mod}"
