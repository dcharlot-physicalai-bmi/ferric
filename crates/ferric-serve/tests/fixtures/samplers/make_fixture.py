"""Reference outputs for ferric-serve's extended samplers, from the code that DEFINES them:
DRY, XTC, top-n-sigma and Mirostat v2 from text-generation-webui modules/sampler_hijack.py @ c93f88712395
(copied verbatim below — p-e-w introduced DRY and XTC there), typical-p from transformers' own
TypicalLogitsWarper (its authors contributed it). Seeded random logits and token contexts.

    python make_fixture.py > reference.json     # torch + transformers
"""
import json, math, random, sys, types
import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import TypicalLogitsWarper
random.seed(20260928); torch.manual_seed(20260928)
NEWLINE, EOS = 7, 3                      # stand-ins for encode("\n")[-1] and eos_token_id
shared = types.SimpleNamespace(tokenizer=types.SimpleNamespace(encode=lambda s: [NEWLINE], eos_token_id=EOS))
def get_device(): return None

class TopNSigmaLogitsWarper(LogitsProcessor):
    def __init__(self, n_sigma: float = 2.0, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        """
        Initialize Top-nσ Sampling logits warper.

        Args:
            n_sigma: The threshold multiplier for standard deviation
            filter_value: Value to assign to filtered logits
            min_tokens_to_keep: Minimum number of tokens to keep
        """
        if n_sigma < 0:
            raise ValueError(f"`n_sigma` must be a non-negative float, but is {n_sigma}")
        self.n_sigma = n_sigma
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Calculate max of logits
        max_logit = torch.max(scores, dim=-1, keepdim=True)[0]

        # Calculate standard deviation only on finite values
        finite_mask = torch.isfinite(scores)
        finite_scores = scores.masked_fill(~finite_mask, 0.0)
        std_logit = torch.std(finite_scores, dim=-1, keepdim=True)

        # Create mask where tokens with logits >= max_logit - n_sigma * std_logit are kept
        threshold = max_logit - self.n_sigma * std_logit
        indices_to_remove = scores < threshold

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep tokens
            top_k_indices = torch.topk(scores, self.min_tokens_to_keep, dim=-1)[1]
            indices_to_remove.scatter_(-1, top_k_indices, False)

        # Apply mask by setting filtered tokens to filter_value
        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        return scores


class XTCLogitsWarper(LogitsProcessor):
    def __init__(self, threshold: float, probability: float, filter_value: float = -float("Inf")):
        self.threshold = threshold
        self.probability = probability
        self.filter_value = filter_value
        self.special_token_ids = [
            shared.tokenizer.encode("\n")[-1],
        ]

        if shared.tokenizer.eos_token_id is not None:
            self.special_token_ids.append(shared.tokenizer.eos_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # `random` returns values in the half-open range [0, 1), so setting `probability`
        # to 0 means the sampler never takes action, while setting it to 1 means the sampler
        # always takes action.
        #
        # Note that while XTC is most intuitively described as "if multiple tokens meet
        # the threshold, then with probability...", reversing the two conditions is logically
        # equivalent, and improves performance because processing can immediately be stopped
        # if the random check fails.
        if random.random() >= self.probability:
            return scores

        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        sorted_indices_to_remove = torch.full_like(probs, False, dtype=torch.bool)

        # This operation sets exactly those indices to `True` for which the next index has
        # probability above the threshold. Since `probs` is sorted, those are the indices
        # of all tokens that meet the threshold, *except* the least probable one.
        sorted_indices_to_remove[..., :-1] = probs[..., 1:] >= self.threshold

        # Convert sorted_indices_to_remove to the original indices
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        # If newline or EOS tokens would be removed, return the original scores
        if indices_to_remove[:, self.special_token_ids].any():
            return scores

        # Otherwise, remove tokens with the mask
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class DRYLogitsProcessor(LogitsProcessor):
    def __init__(self, multiplier: float, base: float, allowed_length: int, sequence_breakers: set[int], _range: int):
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = allowed_length
        self.sequence_breakers = sequence_breakers
        self._range = _range

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._range > 0:
            input_ids = input_ids[:, -self._range:]

        for input_ids_row, scores_row in zip(input_ids, scores):
            # Use normal Python data types for improved performance
            input_ids = input_ids_row.tolist()

            last_token = input_ids[-1]
            if last_token in self.sequence_breakers:
                continue

            # Exclude the last token as it always matches.
            match_indices = []
            for idx, val in enumerate(input_ids[:-1]):
                if val == last_token:
                    match_indices.append(idx)

            # Stores the maximum matching sequence length
            # for each token immediately following the sequence in the input.
            match_lengths = {}

            for i in match_indices:
                next_token = input_ids[i + 1]

                if next_token in self.sequence_breakers:
                    continue

                # We have already found that `last_token` matches at this index,
                # so the match is at least of length 1.
                match_length = 1

                # Extend the match backwards (at most to 50 to prevent exponent overflow at penalty calculation) (this cap also improves performance on worst case)
                while match_length < 50:
                    j = i - match_length
                    if j < 0:
                        # Start of input reached.
                        break

                    previous_token = input_ids[-(match_length + 1)]
                    if input_ids[j] != previous_token:
                        # Start of match reached.
                        break

                    if previous_token in self.sequence_breakers:
                        # Sequence-breaking token reached.
                        break

                    match_length += 1

                if next_token in match_lengths:
                    match_lengths[next_token] = max(match_length, match_lengths[next_token])
                else:
                    match_lengths[next_token] = match_length

            # Apply penalties.
            for token, match_length in match_lengths.items():
                if match_length >= self.allowed_length:
                    penalty = self.multiplier * self.base ** (match_length - self.allowed_length)
                    scores_row[token] -= penalty

        return scores


class MirostatLogitsWarper(LogitsProcessor):
    def __init__(self, mirostat_mode: int, mirostat_tau: float, mirostat_eta: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if mirostat_mode not in [2]:
            raise ValueError(f"`mirostat` has to be a an integer 2, but is {mirostat_mode}")

        self.mirostat_mode = mirostat_mode
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.mu = 2 * self.mirostat_tau
        self.e = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores[0]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()  # candidates

        # Truncate the words with surprise values greater than mu
        for i, candidate in enumerate(prob_original):
            if candidate > 0 and -math.log2(candidate) > self.mu:
                if (i == 0):
                    sorted_logits = sorted_logits[:1]
                else:
                    sorted_logits = sorted_logits[:i]
                break

        # Normalize the probabilities of the remaining words
        prob_topk = torch.softmax(sorted_logits, dim=0)
        prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
        device = get_device()
        if device:
            prob_topk = prob_topk.to(device)
            prev_i = prev_i.to(device)

        observed_surprise = -math.log2(prob_topk[prev_i])
        self.e = observed_surprise - self.mirostat_tau

        # Update mu using the learning rate and error
        self.mu -= self.mirostat_eta * self.e

        sorted_indices_to_remove = torch.ones_like(scores[0], dtype=torch.bool)
        sorted_indices_to_remove[prev_i] = False

        indices_to_remove = sorted_indices_to_remove.unsqueeze(0).scatter(1, sorted_indices.unsqueeze(0), sorted_indices_to_remove.unsqueeze(0))
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


V = 48
def row(): return (torch.randn(1, V) * 2.5).float()
def ctx(n):  # repetitive token contexts so DRY has matches of many lengths
    base = [random.randrange(V) for _ in range(random.randint(3, 9))]
    out = []
    while len(out) < n: out += base[:random.randint(1, len(base))] + [random.randrange(V)]
    return out[:n]
cases = {"dry": [], "xtc": [], "typical": [], "top_n_sigma": [], "mirostat": []}
for _ in range(40):
    r, c = row(), ctx(random.randint(8, 60))
    p = dict(multiplier=random.choice([0.8, 1.0, 2.5]), base=random.choice([1.75, 1.3]), allowed_length=random.choice([1, 2, 3]),
             sequence_breakers=set(random.sample(range(V), 3)), _range=random.choice([0, 16, 32]))
    out = DRYLogitsProcessor(**p)(torch.tensor([c]), r.clone())
    cases["dry"].append({"logits": r[0].tolist(), "context": c, "multiplier": p["multiplier"], "base": p["base"], "allowed_length": p["allowed_length"],
                         "breakers": sorted(p["sequence_breakers"]), "range": p["_range"], "out": out[0].tolist()})
for _ in range(12):   # long verbatim repeats: match lengths of 20-70, so the reference's cap at 50 decides
    r = row(); blk = [random.randrange(V) for _ in range(random.randint(20, 70))]
    c = blk + [random.randrange(V)] + blk[:random.randint(18, len(blk) - 1)]
    p = dict(multiplier=0.8, base=random.choice([1.02, 1.05]), allowed_length=2, sequence_breakers=set(), _range=0)
    out = DRYLogitsProcessor(**p)(torch.tensor([c]), r.clone())
    cases["dry"].append({"logits": r[0].tolist(), "context": c, "multiplier": p["multiplier"], "base": p["base"], "allowed_length": 2,
                         "breakers": [], "range": 0, "out": out[0].tolist()})
for _ in range(40):
    r = row(); th = random.choice([0.02, 0.05, 0.1, 0.2])
    w = XTCLogitsWarper(threshold=th, probability=1.0)            # probability 1: always takes action
    out = w(torch.tensor([[0]]), r.clone())
    cases["xtc"].append({"logits": r[0].tolist(), "threshold": th, "specials": [NEWLINE, EOS], "kept": [bool(x) for x in torch.isfinite(out[0])]})
for _ in range(40):
    r = row(); m = random.choice([0.2, 0.5, 0.9, 0.95])
    out = TypicalLogitsWarper(mass=m)(torch.tensor([[0]]), r.clone())
    cases["typical"].append({"logits": r[0].tolist(), "mass": m, "kept": [bool(x) for x in torch.isfinite(out[0])]})
for _ in range(40):
    r = row()
    if random.random() < 0.5: r[0, random.sample(range(V), 10)] = -float("inf")   # some already removed
    n = random.choice([0.5, 1.0, 2.0])
    out = TopNSigmaLogitsWarper(n_sigma=n)(torch.tensor([[0]]), r.clone())
    cases["top_n_sigma"].append({"logits": [x if math.isfinite(x) else None for x in r[0].tolist()], "n": n, "kept": [bool(x) for x in torch.isfinite(out[0])]})
for _ in range(40):
    r = row(); tau, eta = random.choice([3.0, 5.0]), random.choice([0.1, 0.3])
    w = MirostatLogitsWarper(2, tau, eta)
    steps = []
    for s in range(4):          # four consecutive steps: the truncation count, the draw, and mu after each
        lg = r[0].tolist()
        sl, si = torch.sort(r[0], descending=True); probs = torch.softmax(sl, dim=-1).tolist()
        k = len(probs)
        for i, cnd in enumerate(probs):
            if cnd > 0 and -math.log2(cnd) > w.mu: k = 1 if i == 0 else i; break
        mu_before = w.mu
        out = w(torch.tensor([[0]]), r.clone())
        chosen = int(torch.isfinite(out[0]).nonzero()[0][0])
        steps.append({"logits": lg, "mu_before": mu_before, "k": k, "chosen": chosen, "mu_after": w.mu})
        r = row()
    cases["mirostat"].append({"tau": tau, "eta": eta, "steps": steps})
json.dump({"sources": {"webui": "oobabooga/text-generation-webui c93f88712395 modules/sampler_hijack.py",
                       "transformers": __import__("transformers").__version__, "torch": torch.__version__}, "cases": cases}, sys.stdout)
