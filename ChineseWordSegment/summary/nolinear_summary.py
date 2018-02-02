from summary.text_summary import *
from summary.utils import *
import logging


def get_complete_sentences_with_correlations(sentences, single_subsentence_correlations, text):
    complete_sentences = []
    complete_sentences_correlations = []

    correlations = [None if is_outliner(x, single_subsentence_correlations) else x for x in
                    single_subsentence_correlations]
    single_complete_correlations = [correlations[0]]
    single_complete_sentence = [sentences[0]]

    for index in range(1, len(sentences)):
        sub = sentences[index]
        last_word = single_complete_sentence[-1]
        if in_same_sentence(last_word, sub, text):
            single_complete_sentence.append(sub)
            single_complete_correlations.append(correlations[index])
        else:
            complete_sentences.append(single_complete_sentence)
            complete_sentences_correlations.append(single_complete_correlations)

            single_complete_sentence = [sub]
            single_complete_correlations = [correlations[index]]

        if index == len(sentences) - 1:
            complete_sentences.append(single_complete_sentence)
            complete_sentences_correlations.append(single_complete_correlations)

    return zip(complete_sentences, complete_sentences_correlations)


def get_text_correlation(text):
    text_sentences = get_text_sentence(text)
    distance_map = get_all_sentences_distance(text_sentences)
    distance_sentence_pair = [(string, distance_map[string]) for string in text_sentences]
    correlation = [1 - d for _, d in distance_sentence_pair]
    segments_with_index = [index_word for index_word in enumerate(text_sentences)]
    return segments_with_index, correlation, distance_sentence_pair


def get_one_file_complex_correlation(text, title):
    logging.debug("{} {}".format(text[:50], title))

    correlations = get_text_correlation(text)
    sentences = [s for s, d in correlations[2]]
    title_distance = get_title_distance(title, sentences)
    title_correlations = softmax([1 - d for _, d in title_distance])
    content_correlations = softmax(correlations[1])
    complex_correlation = get_complex_correlation(title_correlations, content_correlations)
    # plot_correlation(complex_correlation[0])
    return complex_correlation


def get_merged_correlation(single_nolinear_correlations, sentences, f):
    merged_correlation = f(single_nolinear_correlations, sentences)
    return merged_correlation


def f(array, sentences):
    total_words_length = len("".join(sentences))
    array = list(filter(lambda x: x is not None, array))
    content_ratio = len(array) / len(sentences)
    result = np.mean(array) * ((np.log(total_words_length * content_ratio + 1)))
    return result if not np.isnan(result) else -1


def get_summary_with_nolinear(text, title, fit_length):
    complex_correlation = get_one_file_complex_correlation(text, title)
    sentences = get_text_sentence(text)
    complete_nolinear = get_complete_sentences_with_correlations(
        sentences, complex_correlation, get_text_content(text, escape_english=False))

    complete_nolinear = list(complete_nolinear)

    def get_sentence_and_merged_correlation(subsentences, correlation):
        return (" ".join(subsentences), correlation)

    completed_sentences_with_correlations = []
    all_correlations = [c for s, c in complete_nolinear]
    #    _25th_all_correlations = np.

    for s, c in complete_nolinear:
        merged_correlation = get_merged_correlation(c, s, f)
        single_completed_sentence_with_correlation = get_sentence_and_merged_correlation(s, merged_correlation)
        completed_sentences_with_correlations.append(single_completed_sentence_with_correlation)

    # _25_percentile = np.percentile([c for s, c in completed_sentences_with_correlations], 25)
    #    _60_percentile = np.percentile([c for s, c in completed_sentences_with_correlations], 60)
    correlations = [c for s, c in completed_sentences_with_correlations]
    correlations = [x if not np.isnan(x) else -1 for x in correlations]
    sentences = [s for s, c in completed_sentences_with_correlations]

    top_correlations = top_n(correlations, sentences, fit_length)

    total_sentence = []
    total_length = 0
    min_single_length = 3
    for string, correlations in zip(sentences, correlations):
        if correlations in top_correlations:
            if len(string) >= min_single_length:
                total_length += len(string)
                total_sentence.append(string)

    return "。".join(total_sentence)


def get_suitable_length_summary(text, title, fit_length):
    summary = get_summary_with_nolinear(text, title, fit_length)
    return summary


def readable_summary(text, title):
    fit_length = get_fit_length(len(get_text_content(text)))
    return title + ": " + get_suitable_length_summary(text, title, fit_length)



if __name__ == '__main__':
    target_file_path = '../experiment/error_analysis.txt'
    title = '腾讯工地着火新浪保安伸援手'
    summary = readable_summary(target_file_path, title)
    # print(summary)
    print(recovery_punctuation(summary, title + ":\n" + get_text_content(target_file_path, escape_english=False)))

