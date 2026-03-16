# tonecraft 
is a python library that genenerates synthetic coversatational data for tone and speech pattern training for SLM fine-tunning.

# Context

It follows a question-answer setup. A questioner with a role assigned (ex. restaurant client, store client) and a responder (ex. waiter, store consultant).
It generates the data in json formatted files.

# Stack
 * python
 * instructor
 * pydantic
 * pytest


# Flow 
1. package is triggered via a standartized .md format that provides context  about the domain.
2. The context is used to create a domain expert agent. It researches the domain. Provides a schema for balanced data distribution (must be a top notch thinking model.
3. The domain expert provides background and instructuions to the questioner and responder
agents
4. After evalution of the respones, low confidence response are dropped and if needed  
more iterations are done to rebalance the data distribution. (number of iterations can be set)