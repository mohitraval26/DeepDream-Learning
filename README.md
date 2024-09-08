# DeepDream Learning

**DeepDream Learning** uses Natural Language Processing (NLP) and Machine Learning (ML) to analyze and classify dreams based on emotional content. Input your dream, and the tool identifies emotions such as anger, joy, and fear, providing insights into your dream's emotional content. This project leverages data from DreamBank and advanced text analysis techniques to offer a deeper understanding of your subconscious mind.

## Features

- **Text Preprocessing:** Cleans and tokenizes dream text to prepare for analysis.
- **Topic Modeling:** Uses Latent Dirichlet Allocation (LDA) to extract topics from dream content.
- **Sentiment Analysis:** Analyzes sentiment using VADER to capture emotional tones.
- **Emotion Classification:** Classifies dreams into various emotional categories using a RandomForestClassifier.
- **Interactive Analysis:** Provides emotional insights based on user-inputted dream descriptions.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sompuradhruv/DeepDream-Learning.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd DeepDream-Learning
    ```

3. **Install Python (if not already installed):**

    Ensure you have Python 3.6 or later installed. You can download it from [python.org](https://www.python.org/downloads/).

4. **Install the required packages:**

    Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

    Install the necessary Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

5. **Prepare the dataset:**

    Unzip the dataset file to use it in its CSV format.

## Usage

1. **Run the script:**

    ```bash
    python code.py
    ```

2. **Input your dream description** when prompted and receive emotional insights.

## Example

Here is how you can use the `classify_dream` function:

```python
user_input = "I had a dream where I was flying over a beautiful landscape."
dream_features = classify_dream(user_input)

print("Dream features:")
for emotion, value in dream_features.items():
    print(f"{emotion}: {value}")
```

## Contributing

Feel free to open issues or submit pull requests to enhance the functionality or improve the analysis methods.

---
