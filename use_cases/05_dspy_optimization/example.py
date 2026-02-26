import csv
import dspy
from silicon_refinery.dspy_ext import AppleFMLM


# A DSPy module for generating summaries and classifying
class SupportClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        # Chain of Thought prompts the model to think step-by-step
        self.analyze = dspy.ChainOfThought("customer_email -> summary, priority")

    def forward(self, email):
        return self.analyze(customer_email=email)


def main():
    # 1. Initialize our custom local model provider
    local_lm = AppleFMLM()

    # 2. Configure DSPy to use the Apple Foundation Model
    dspy.settings.configure(lm=local_lm)

    # 3. Use the pipeline
    classifier = SupportClassifier()

    print("Evaluating DSPy Chain of Thought on local Neural Engine using support_tickets.csv...\n")

    with open("datasets/support_tickets.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            email = row["email_body"]
            print(f"Ticket: {row['ticket_id']} | Subject: {row['email_subject']}")

            # Predict
            result = classifier(email)

            print(f"DSPy Summary: {result.summary}")
            print(f"DSPy Priority: {result.priority}\n")


if __name__ == "__main__":
    main()
