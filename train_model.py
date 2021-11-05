import torch
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import  BertTokenizer, LineByLineTextDataset, TrainingArguments, Trainer


bert_file = "bert-base-uncased"

config = BertConfig.from_pretrained(bert_file)
tokenizer = BertTokenizer.from_pretrained(bert_file)
model = BertForMaskedLM.from_pretrained(bert_file)
print('No of parameters: ', model.num_parameters())

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=save_path, block_size=512)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
print('No. of lines: ', len(dataset))
!mkdir outputs
training_args = TrainingArguments(
    output_dir='./outputs/',
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=16,
    save_steps=5000,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model('./outputs/')
