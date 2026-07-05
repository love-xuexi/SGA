import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
device = "cuda:5"
path = "path/to/ArmoRM-Llama3-8B-v0_1"
model = AutoModelForSequenceClassification.from_pretrained(path, device_map=device, 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)



# We load a random sample from the validation set of the HelpSteer dataset
prompt = """Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea? Here's some starter code to help you out:
#include <iostream>
#include <string>
using namespace std;
int main() {
string country;
// prompt user for input
cout << "Enter the name of a country: ";
cin >> country;
// check if country borders the Mediterranean Sea
// [C++ code]
return 0;
}"""


response = """int main() {
string country;
// prompt user for input
cout << "Enter the name of a country: ";
cin >> country;
// check if country borders the Mediterranean Sea
if (endsWith(country, "Mediterranean")) {
cout << "Yes, the country " << country
<< " borders the Mediterranean Sea.";
} else {
cout << "No, the country " << country
<< " does not border the Mediterranean Sea.";
}
return 0;
}"""
def clean_prompt(prompt: str) -> str:
    """
    Clean special formatting from the prompt and extract the clean question content.
    Supports cleaning the following three formats:
    1. GAM format: "Your response must satisfy the following scores: helpfulness_score: 0.50 and ..."
    2. ric format: "<rm1_score> 0.691 <rm2_score> 0.693 ..."
    3. cpo format: "< helpfulness: 5 > < correctness: 5 > ..."
    """
    if not prompt:
        return ""
    
    cleaned = prompt
    
    # 1. GAM format: Match "Your response must satisfy the following scores: xxx_score: x.xx and yyy_score: x.xx ..."
    # Pattern: "Your response must satisfy the following scores:" followed by multiple "xxx_score: number" connected by "and"
    gam_pattern = r"Your response must satisfy the following scores:\s*(?:\w+_score:\s*[\d.]+(?:\s+and\s+)?)+\s*"
    cleaned = re.sub(gam_pattern, "", cleaned, flags=re.IGNORECASE)
    
    # 2. ric format: Match patterns like "<rm1_score> 0.691 <rm2_score> 0.693 ..."
    # Pattern: Multiple "<rmN_score> number" or similar formats
    ric_pattern = r"(?:<\w+_score>\s*[\d.]+\s*)+"
    cleaned = re.sub(ric_pattern, "", cleaned, flags=re.IGNORECASE)
    
    # 3. cpo format: Match patterns like "< helpfulness: 5 > < correctness: 5 > ..."
    # Pattern: Multiple "< xxx: number >" formats
    cpo_pattern = r"(?:<\s*\w+:\s*[\d.]+\s*>\s*)+"
    cleaned = re.sub(cpo_pattern, "", cleaned, flags=re.IGNORECASE)
    
    # Clean the leading "Human: " or "H: " prefix (keep the question content)
    cleaned = re.sub(r"^H:\s*", "", cleaned.strip())
    
    # Clean the trailing "\n\nA:" or "\nA:" or "A:" suffix
    cleaned = re.sub(r"\s*\n*A:\s*$", "", cleaned.strip())
    
    # Clean up redundant whitespace characters
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned
    
messages = [{"role": "user", "content": prompt},
           {"role": "assistant", "content": response}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
with torch.no_grad():
   output = model(input_ids)
   # Multi-objective rewards for the response
   multi_obj_rewards = output.rewards.cpu().float() 
   # The gating layer's output is conditioned on the prompt
   gating_output = output.gating_output.cpu().float()
   # The preference score for the response, aggregated from the 
   # multi-objective rewards with the gating layer
   preference_score = output.score.cpu().float()  
# We apply a transformation matrix to the multi-objective rewards
# before multiplying with the gating layer's output. This mainly aims
# at reducing the verbosity bias of the original reward objectives
obj_transform = model.reward_transform_matrix.data.cpu().float()
# The final coefficients assigned to each reward objective
multi_obj_coeffs = gating_output @ obj_transform.T
# The preference score is the linear combination of the multi-objective rewards with
# the multi-objective coefficients, which can be verified by the following assertion
assert torch.isclose(torch.sum(multi_obj_rewards * multi_obj_coeffs, dim=1), preference_score, atol=1e-3) 
# Find the top-K reward objectives with coefficients of the highest magnitude
K = 3
top_obj_dims = torch.argsort(torch.abs(multi_obj_coeffs), dim=1, descending=True,)[:, :K]
top_obj_coeffs = torch.gather(multi_obj_coeffs, dim=1, index=top_obj_dims)

# The attributes of the 19 reward objectives
attributes = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
   'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
   'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
   'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
   'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
   'code-style','code-explanation','code-instruction-following','code-readability']

example_index = 0
for i in range(K):
   attribute = attributes[top_obj_dims[example_index, i].item()]
   coeff = top_obj_coeffs[example_index, i].item()
   print(f"{attribute}: {round(coeff,5)}")
# code-complexity: 0.19922
# helpsteer-verbosity: -0.10864
# ultrafeedback-instruction_following: 0.07861

# The actual rewards of this example from the HelpSteer dataset
# are [3,3,4,2,2] for the five helpsteer objectives: 
# helpfulness, correctness, coherence, complexity, verbosity
# We can linearly transform our predicted rewards to the 
# original reward space to compare with the ground truth
print(multi_obj_rewards)
helpsteer_rewards_pred = multi_obj_rewards[0, :5] * 5 - 0.5
ultrafeedback_reward_pred = multi_obj_rewards[0,5:10] * 6.25 - 0.625

print(helpsteer_rewards_pred)
print(ultrafeedback_reward_pred)
# [2.78125   2.859375  3.484375  1.3847656 1.296875 ]