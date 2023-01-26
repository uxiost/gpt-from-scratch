# GPT from scratch
GPT implementation following Karpathy's lesson https://www.youtube.com/watch?v=kCc8FmEb1nY

# Setup
You only need a GPU and pytorch. The easiest way is to [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uxiost/gpt-from-scratch/blob/main/notebooks/gpt_main.ipynb) Remember to select the correct runtime (Runtime -> Change runtime type -> GPU)

Locally you can:  
- Clone repo
- Install requirements > pip install -r requirements.txt
- Run > python gpt.py

# Output sample
After training for 5000 steps in the Shakespeare corpus, we reach a validation loss around 1.49. Generated samples look like this:

```
> WARWICK:
No, I fair and the cause I see by grieving grief:
Thy after glass becomived. look, what the conciful
My sound brow still this kingdrings, that he  is castled
That she foul the maidenh; commens with eagl in his famise,
And would over-hig, bid let them arre they do;
Now 'twere no daughter, for, for a while,
By grief thit you now will another; I have reported: alabiant the country
Where I do undertny on my tongue at his profound,
Were here same nor her: I long in the rood of it.

STANLEY:
Madam, I heard the wife,
And then greeting, give me for conclured,
Caming arms the lives-gace; and migh would strike it.

JULIET:
I am no sall of eye. Twelcome a prove,
Return no received to speak a brieaty and
On one tender bothers at I keep that heart; honour
ye with her desert; and do are you away.

GLOUCESTER:
God, then, girl; and say King Lewis manymen:
My boy light are command now to London the rottemphs:
Plantagenet, what Bianca directions,
And did fetch my soldiers; my love of peace
Which with honour coming suss dishonourse,
And slaughter his despate; and ha?
```

# Suggested exercises
Suggested exercises:
- EX1: The n-dimensional tensor mastery challenge: Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension (answer is in nanoGPT).
- EX2: Train the GPT on your own dataset of choice! What other data could be fun to blabber on about? (A fun advanced suggestion if you like: train a GPT to do addition of two numbers, i.e. a+b=c. You may find it helpful to predict the digits of c in reverse order, as the typical addition algorithm (that you're hoping it learns) would proceed right to left as it adds the numbers, keeping track of a carry along the way. You may want to modify the data loader to simply serve random problems and skip the generation of train.bin, val.bin. You may want to mask out the loss at the input positions of a+b that just specify the problem using y=-1 in the targets (see CrossEntropyLoss ignore_index). Does your Transformer learn to add? Especially on a validation set of addition problems it hasn't seen during training? Once you have this, swole doge project: build a calculator clone in GPT, for all of +-*/. Not an easy problem. You may need Chain of Thought traces.)
- EX3: Find a dataset that is very large, so large that you can't see a gap between train and val loss. Pretrain the transformer on this data, then initialize with that model and finetune it on tiny shakespeare with a smaller number of steps and lower learning rate. Can you obtain a lower validation loss by the use of pretraining?
- EX4: Read some transformer papers and implement one additional feature or change that people seem to use. Does it improve the performance of your GPT?

# TODO
[] Exercises  
[] Google Colab runner  
[] Video transcription/summary  
