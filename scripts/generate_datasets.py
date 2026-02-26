import csv
import os


def create_medical_notes():
    data = [
        ["id", "date", "raw_note"],
        [
            1,
            "2023-10-01",
            "Patient complains of severe chest pain and shortness of breath for the last 2 hours. Requires immediate attention.",
        ],
        [
            2,
            "2023-10-02",
            "Routine checkup. Patient reports mild fatigue but otherwise healthy. Prescribed rest for 3 days.",
        ],
        [
            3,
            "2023-10-03",
            "Patient fell from ladder. Sharp pain in right forearm and visible swelling. Possible fracture. Wait time 1 day.",
        ],
        [
            4,
            "2023-10-04",
            "Experiencing chronic migraines accompanied by nausea. Occurs roughly twice a week for the last 14 days.",
        ],
    ]
    with open("datasets/medical_notes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def create_support_tickets():
    data = [
        ["ticket_id", "email_subject", "email_body"],
        [
            "T-100",
            "Cannot login",
            "I've been trying to reset my password but the link never arrives. Please help me access my account!",
        ],
        [
            "T-101",
            "Billing issue",
            "I was charged twice this month for my premium subscription. I need a refund immediately.",
        ],
        [
            "T-102",
            "Feature request",
            "It would be great if we could export our dashboard data to a PDF format instead of just CSV.",
        ],
        [
            "T-103",
            "System down",
            "The entire production database seems to be offline. All our users are seeing 500 errors!",
        ],
    ]
    with open("datasets/support_tickets.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def create_product_reviews():
    data = [
        ["review_id", "product", "review_text"],
        [
            "R1",
            "SmartWatch Pro",
            "The battery life is amazing, easily lasts a week. However, the screen is a bit dim in direct sunlight.",
        ],
        [
            "R2",
            "NoiseCancelling Headphones",
            "Terrible audio quality. The bass is non-existent and the noise cancellation barely works. Returning it.",
        ],
        [
            "R3",
            "Ergonomic Keyboard",
            "It gets the job done. Comfortable to type on but the keys are a bit too loud for an open office.",
        ],
        [
            "R4",
            "4K Monitor",
            "Absolutely stunning visuals! The color accuracy is perfect for my video editing workflow.",
        ],
    ]
    with open("datasets/product_reviews.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def create_server_logs():
    data = [
        ["log_id", "timestamp", "log_message"],
        [
            "L1",
            "2023-10-01T10:00:00Z",
            "10:00 AM [auth_service] ERROR - Failed to validate token for user 123. Invalid signature.",
        ],
        [
            "L2",
            "2023-10-01T10:01:00Z",
            "10:01 AM [db_connector] INFO - Connection established successfully to primary cluster.",
        ],
        [
            "L3",
            "2023-10-01T10:02:00Z",
            "10:02 AM [api_gateway] WARNING - High latency detected on route /users. Avg response time 1.5s.",
        ],
        [
            "L4",
            "2023-10-01T10:03:00Z",
            "10:03 AM [payment_service] ERROR - Transaction declined for cart 8921. Insufficient funds.",
        ],
    ]
    with open("datasets/server_logs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    create_medical_notes()
    create_support_tickets()
    create_product_reviews()
    create_server_logs()
    print("Real-world CSV datasets generated successfully.")
