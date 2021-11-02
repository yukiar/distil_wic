from nltk.stem import SnowballStemmer
from sklearn.preprocessing import normalize
import numpy as np
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

def smart_lower(word, do_lower=False):
    if do_lower:
        return word.lower()
    elif str.istitle(word):
        return word.lower()
    else:
        return word


def standardize_matrix(vecs):
    # Normalize and mean-center
    mean = np.mean(vecs, axis=1, keepdims=True)
    svecs = vecs - np.tile(mean, (1, vecs.shape[1]))
    svecs = normalize(svecs, norm='l2')

    return svecs


def _run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    uniname = unicodedata.name(char, '###NONAME###').lower()
    if "hyphen" in uniname or "dash" in uniname:  # do not separate hyphens
        return False
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _basic_tokenization(sentences, do_lower_case):
    tokenized_sents = []
    for sent in sentences:
        split_tokens = []
        for token in sent.split(' '):
            if do_lower_case:
                token = token.lower()
            split_tokens.extend(_run_split_on_punc(token))

        tokenized_sents.append(" ".join(split_tokens))

    return tokenized_sents


def tokenize(tokenizer, space_symbol, subword_symbol, padding, max_length, truncation, match_all, ambiguous_match, term,
             sentences,
             logger=None):
    if subword_symbol == '##':
        # Needs manual lower() for BERT talkenizer to set "do_basic_tokenize=False"
        sentences = _basic_tokenization(sentences, tokenizer.init_kwargs['do_lower_case'])

    # Tokenize input
    batch_encoding = tokenizer(
        sentences,
        padding=padding,
        max_length=max_length,
        truncation=truncation,
    )
    lowercase_setting = tokenizer.init_kwargs['do_lower_case'] if 'do_lower_case' in tokenizer.init_kwargs else False
    input_ids, token_type_ids, attention_mask = [], [], []
    for i in range(len(sentences)):
        for k in batch_encoding:
            eval(k).append(batch_encoding[k][i])

    term_idxes = []
    stemmer = SnowballStemmer('english') if ambiguous_match else None
    for i, ids in enumerate(input_ids):
        match_target = smart_lower(stemmer.stem(term), lowercase_setting) if ambiguous_match \
            else smart_lower(term, lowercase_setting)
        if space_symbol is not None:
            term_idx = find_term_idx_with_spacesymbol(tokenizer, stemmer, space_symbol, match_target, ids, match_all)
        else:
            term_idx = find_term_idx_with_subwordsymbol(tokenizer, stemmer, subword_symbol, match_target, ids,
                                                        match_all)

        if 1 not in term_idx:
            term_idx = fallback_to_subsequence_matching(tokenizer, stemmer, space_symbol, subword_symbol, match_target, ids)

        term_idxes.append(term_idx)

    if logger is not None:
        for i, example in enumerate(sentences[:5]):
            logger.info("*** Example ***")
            logger.info("sentence: %s" % (example))
            logger.info("input_ids: %s" % input_ids[i])

    return input_ids, attention_mask, term_idxes


def findLongestSequence(A, k):
    # Function to find the maximum sequence of continuous 1's by replacing
    # at most `k` zeroes by 1 using sliding window technique

    left = 0  # represents the current window's starting index
    count = 0  # stores the total number of zeros in the current window
    window = 0  # stores the maximum number of continuous 1's found
    # so far (including `k` zeroes)

    leftIndex = 0  # stores the left index of maximum window found so far

    # maintain a window `[left…right]` containing at most `k` zeroes
    for right in range(len(A)):

        # if the current element is 0, increase the count of zeros in the
        # current window by 1
        if A[right] == 0:
            count = count + 1

        # the window becomes unstable if the total number of zeros in it becomes
        # more than `k`
        while count > k:
            # if we have found zero, decrement the number of zeros in the
            # current window by 1
            if A[left] == 0:
                count = count - 1

            # remove elements from the window's left side till the window
            # becomes stable again
            left = left + 1

        # when we reach here, window `[left…right]` contains at most
        # `k` zeroes, and we update max window size and leftmost index
        # of the window
        if right - left + 1 > window:
            window = right - left + 1
            leftIndex = left

    # if window == 0, no sequence found
    # print the maximum sequence of continuous 1's
    # print("The longest sequence has length", window, "from index",
    #       leftIndex, "to", (leftIndex + window - 1))

    return window, (leftIndex, leftIndex + window - 1)


def fallback_to_subsequence_matching(tokenizer, stemmer, space_symbol, subword_symbol, term, token_ids):
    filtered_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    # Remove subword/space symbol
    if space_symbol is not None:
        filtered_tokens = [w.replace(space_symbol, '') for w in filtered_tokens]
    if subword_symbol is not None:
        filtered_tokens = [w.replace(subword_symbol, '') for w in filtered_tokens]

    # Stemming if desired
    if stemmer is not None:
        filtered_tokens = [stemmer.stem(w) for w in filtered_tokens]

    term_idxes = [0] * len(token_ids)
    for idx, token in enumerate(filtered_tokens):
        if token in term:
            term_idxes[idx] = 1

    if 1 in term_idxes:
        window, span = findLongestSequence(term_idxes, 0)
        term_idxes = [0] * len(token_ids)
        if window > 0:  # the longest sequence found
            for i in range(span[0], span[1] + 1):
                term_idxes[i] = 1
    return term_idxes


def find_term_idx_with_subwordsymbol(tokenizer, stemmer, subword_symbol, term, token_ids, match_all):
    filtered_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    current_sub_text = []
    term_idxes = [0] * len(token_ids)
    match_found = False
    for idx, token in enumerate(filtered_tokens):
        if token in tokenizer.all_special_tokens:
            continue
        if token.startswith(subword_symbol):  # subword
            current_sub_text.append(token)
        else:
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text).strip()
                if stemmer is not None:
                    sub_text = stemmer.stem(sub_text)
                sub_text = smart_lower(sub_text)
                if term == sub_text:
                    for i in range(len(current_sub_text)):
                        term_idxes[idx - 1 - i] = 1
                    if not match_all:
                        match_found = True
                        break
                current_sub_text = []
            current_sub_text.append(token)

    if not match_all and match_found:
        pass
    else:
        if current_sub_text:
            sub_text = tokenizer.convert_tokens_to_string(current_sub_text).strip()
            if stemmer is not None:
                sub_text = stemmer.stem(sub_text)
            sub_text = smart_lower(sub_text)
            if term == sub_text:
                for i in range(len(current_sub_text)):
                    term_idxes[len(filtered_tokens) - 1 - i] = 1

    return term_idxes


def find_term_idx_with_spacesymbol(tokenizer, stemmer, space_symbol, term, token_ids, match_all):
    filtered_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    current_sub_text = []
    term_idxes = [0] * len(token_ids)
    match_found = False
    for idx, token in enumerate(filtered_tokens):
        if token in tokenizer.all_special_tokens:
            continue
        if token.startswith(space_symbol):
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text).strip()
                if stemmer is not None:
                    sub_text = stemmer.stem(sub_text)
                sub_text = smart_lower(sub_text)
                if term == sub_text:
                    if space_symbol in current_sub_text:
                        current_sub_text.remove(
                            space_symbol)  # Chinese tokenization often outputs only space symbols...
                    for i in range(len(current_sub_text)):
                        term_idxes[idx - 1 - i] = 1
                    if not match_all:
                        match_found = True
                        break
                current_sub_text = []
            current_sub_text.append(token)
        else:  # subword
            current_sub_text.append(token)

    if not match_all and match_found:
        pass
    else:
        if current_sub_text:
            sub_text = tokenizer.convert_tokens_to_string(current_sub_text).strip()
            if stemmer is not None:
                sub_text = stemmer.stem(sub_text)
            sub_text = smart_lower(sub_text)
            if term == sub_text:
                for i in range(len(current_sub_text)):
                    term_idxes[len(filtered_tokens) - 1 - i] = 1

    return term_idxes

##### For Evaluation ####
def sort_by_idx(item_dict):
    ranked_list = []
    sorted_list = sorted(item_dict.items(), key=lambda x: x[0])

    for key, val in sorted_list:
        ranked_list.append(val)

    return ranked_list

def Cosine(matrix_x, matrix_y):
    mean_x = matrix_x[1:-1].mean(axis=0).unsqueeze(0).cpu().numpy()  # Skip [CLS] and [SEP]
    mean_y = matrix_y[1:-1].mean(axis=0).unsqueeze(0).cpu().numpy()  # Skip [CLS] and [SEP]

    mean_cos = cosine_similarity(mean_x, mean_y)[0, 0]

    return mean_cos

def compute_accuracy(gold_dict, predicted_dict):
    assert len(gold_dict) == len(predicted_dict)
    if len(gold_dict) == 0 or len(predicted_dict) == 0:
        return 0.
    gold_vals, predicted_vals = [], []
    for key in gold_dict.keys():
        gold_vals.append(gold_dict[key])
        predicted_vals.append(predicted_dict[key])

    return accuracy_score(gold_vals, predicted_vals)