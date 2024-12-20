import os
import re
import random
from tqdm import tqdm
import gzip
import json
from openai import OpenAI

client = OpenAI(
    # api_key=
    # base_url=
)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 3
ICL_FLAG = False
COT_FLAG = False
REFLEXION_FLAG = True
ANSWER_TRIGGER = "The answer is"

def answer_question(question):
    """ completion = client.chat.completions.create(
        model = "moonshot-v1-8k",
        messages = [
            {"role": "system", "content": "You are an AI math assistant. You provide safe, helpful, and accurate answers to users."},
            {"role": "user", "content": question},
        ],
        temperature = 0.3,
    ) """
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages = [
            {"role": "system", "content": "You are an AI math assistant. You provide safe, helpful, and accurate answers to users."},
            {"role": "user", "content": question},
        ],
        stream=False
    )
    return completion.choices[0].message.content

def load_jsonl(file_path, instruction="instruction", input="input", output="output", category="category", is_gzip=False):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None,
            )
            item = new_item
            list_data_dict.append(item)
    return list_data_dict

def Reflexion(input_text, env_answer, max_reflexion_times=3):
    not_solved = True
    reflexion_times = 0
    while reflexion_times < max_reflexion_times and not_solved:
        answer = answer_question(input_text)
        simple_answer = clean_answer(answer)
        not_solved = not is_correct(simple_answer, env_answer)
        if not_solved:
            input_text = input_text + " An incorrect answer is " + answer + ". Please try again, A:"
        else:
            break
        reflexion_times += 1
    return not_solved

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def create_demo_text(n_shot=8):
    question, chain, answer, reflection = [], [], [], []
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if COT_FLAG:
            demo_text += ("Q: "+question[i]+"\nA: "+chain[i]+" "+ANSWER_TRIGGER+" "+answer[i]+".\n\n")
        elif ICL_FLAG:
            demo_text += ("Question: "+question[i]+"\nAnswer: "+ANSWER_TRIGGER+" "+answer[i]+".\n\n")
        else:
            demo_text = ""
    return demo_text

def build_prompt(input_text, n_shot):
    demo = create_demo_text(n_shot)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def main():
    test_filepath = "./gsm8k/test.jsonl"
    list_data_dict = load_jsonl(test_filepath, instruction="question", output="answer")

    answers = []
    print("Start testing ...")
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample["instruction"], N_SHOT)
        if REFLEXION_FLAG:
            if_not_solved = Reflexion(input_text, sample["output"])
            is_cor = not if_not_solved
            print("reflexion")
        else:
            model_completion = answer_question(input_text)
            model_answer = clean_answer(model_completion)
            is_cor = is_correct(model_answer, sample["output"])
        answers.append(is_cor)
        # print(f"Full input_text:\n{input_text}\n\n")
        """ print(
            f'Question: {sample["instruction"]}\n\n'
            f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
            f"Model Answers: {model_answer}\n\n"
            f"Model Completion: {model_completion}\n\n"
            f"Is correct: {is_cor}\n\n"
        ) """

        """ print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}."
        ) """

    os.makedirs("results", exist_ok=True)

    with open("./results/answers.txt", "w") as f:
        for answer in answers:
            print(answer, file=f)

    with open("./results/accuracy.txt", "w") as f:
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}.",
            file=f,
        )

if __name__ == "__main__":
    main()