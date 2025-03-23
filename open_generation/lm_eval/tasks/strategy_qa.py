import re

from evaluate import load

from lm_eval.base import Task

import pandas as pd
class StrategyQA(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "data/strategy_qa"

    def __init__(self, postprocessed_output_path, sft):
        self.postprocessed_output_path = postprocessed_output_path
        self.sft = sft
        super().__init__(
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of commonsense_qa can be loaded with old datasets cache
        assert (
            len(dataset) == 2286
        ), "please ensure you have the latest version of commonsense_qa dataset, try deleting its old cache"
        return dataset

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "".join(doc["label"])

    def postprocess_generation(self, generation):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        completion = generation[0].lower().strip()
        answer_key = "None"
        if self.sft:
            answer_begin_hints = ["answer is", "answer to", "answer choice is", "i would choose", "answer would be", "answer seems to be", "the correct answer is", "answer to the question is"]
            answer_end_hints = ["is the correct answer", "is the correct choice", "is correct", "is the best choice", "answer choice is correct"]
            error = False
            completion = generation[0].replace("\n\n", "\n").replace(":", " ").strip().lower() + "\n"
            lines = completion.split("\n")
            lines = lines[::-1]
            for line_idx in range(len(lines)):
                line = lines[line_idx]
                matched_begin_hints = [_ for _ in answer_begin_hints if _ in line]
                matched_end_hints = [_ for _ in answer_end_hints if _ in line]
                if len(matched_begin_hints) > 0 and len(matched_end_hints) > 0:
                    print("=" * 25 + "Too much matched hints" + "=" * 25)
                    error = True
                elif len(matched_begin_hints) == 0 and len(matched_end_hints) == 0:
                    continue
                elif len(matched_begin_hints) == 1:
                    completion = line.split(matched_begin_hints[0])[1]
                    if len(completion) < 3 and line_idx + 2 < len(lines):
                        completion = completion + lines[line_idx + 1]
                elif len(matched_end_hints) == 1:
                    completion = line.split(matched_end_hints[0])[0]
                    
                pattern = r'\(([abcde])\)'
                matches = re.findall(pattern, completion)
                if matches:
                    answer_key = matches[-1].upper()
                    break
                else:
                    completion = generation[0].replace("\n\n", "\n").strip().lower() + "\n"
                    continue   

            if answer_key != "None":
                return "yes" if answer_key == "A" else "no"

            if len(matched_begin_hints) == 0 and len(matched_end_hints) == 0:
                print("=" * 25 + "No matched hint" + "=" * 25)
                error = True

            pattern = r'\(([abcde])\)'
            matches = re.findall(pattern, completion)
            if matches:
                answer_key = matches[-1].upper()
            else:
                print("=" * 25 + "No matched results" + "=" * 25)
                error = True
            if error:
                print(answer_key)
                print(generation[0])
                print("=" * 120 + "\n\n")
            return "yes" if answer_key == "A" else "no"

        else:
            if "\n\n" in completion:
                completion = completion.split("\n\n")[0]
            completion = generation[0].lower().strip()
            answer_keys = ["yes", "no"]
            if "he answer is " in completion:
                completion = completion.split("he answer is ")[-1]
                completion = completion[:3]
            else:
                completion = " " + completion.replace(".", " ").replace(",", " ").replace(";", " ")
                answer_keys = [" yes ", " no "]

            matched_keys = [key.strip() for key in answer_keys if key in completion]

            if len(matched_keys) == 1:
                answer_key = matched_keys[0]
            else:
                print("=" * 25 + "No clear yes or no results" + "=" * 25 )
                print(generation[0])
                print("=" * 50 + "=" * len("No clear yes or no results") + "\n")

        return answer_key


    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        generations = [self.postprocess_generation(_) for _ in generations]
        if self.postprocessed_output_path:
            postprocessed_output = pd.DataFrame()
            postprocessed_output['results'] = generations
            postprocessed_output.to_json(self.postprocessed_output_path, orient='records', lines=True)
        cnt = 0
        for i in range(len(generations)):
            if generations[i] == "None":
                cnt += 1
        acc_metric = load("exact_match")
        results = acc_metric.compute(
            references=references,
            predictions=generations,
        ) 
        results["match_template"] = 1 - cnt / len(generations)
        return results
