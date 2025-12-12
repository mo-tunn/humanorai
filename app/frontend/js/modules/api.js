export class ApiService {
    constructor(baseUrl = 'http://127.0.0.1:8000') {
        this.baseUrl = baseUrl;
    }

    async predict(text) {
        try {
            const response = await fetch(`${this.baseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Server error occurred.');
            }

            return await response.json();

        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    async predictFile(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.baseUrl}/predict_file`, {
                method: 'POST',
                // Content-Type header is not set manually for FormData, browser sets it with boundary
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Server error occurred.');
            }

            return await response.json();

        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    getMockData() {
        console.warn("Backend unavailable, using mock data.");
        return {
            "individual_results": [
                {
                    "model": "Logistic Regression",
                    "probability_percent": 3.45,
                    "decision": "DEFINITELY HUMAN"
                },
                {
                    "model": "Random Forest",
                    "probability_percent": 12.50,
                    "decision": "LIKELY HUMAN"
                },
                {
                    "model": "Neural Network",
                    "probability_percent": 88.20,
                    "decision": "DEFINITELY AI"
                },
                {
                    "model": "AdaBoost",
                    "probability_percent": 45.00,
                    "decision": "UNCERTAIN"
                }
            ],
            "ensemble_average_percent": 37.29,
            "ensemble_decision": "UNCERTAIN",
            "input_length": 500
        };
    }
}
