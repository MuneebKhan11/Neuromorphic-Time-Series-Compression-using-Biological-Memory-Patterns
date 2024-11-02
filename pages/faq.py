import streamlit as st


def apply_custom_styles():
    st.markdown("""
        <style>
        /* Base styles */
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Header styles */
        .faq-header {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .faq-header h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }

        .faq-header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        /* FAQ section styles */
        .faq-section {
            max-width: 800px;
            margin: 0 auto;
            padding: 1rem;
        }

        /* Navigation styles */
        .navigation {
            text-align: center;
            margin-top: 3rem;
        }

        .back-button {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #667eea;
            color: white;
            border-radius: 5px;
            text-decoration: none;
            transition: background 0.3s ease;
        }

        .back-button:hover {
            background: #5a67d8;
        }

        /* Footer styles */
        .footer {
            text-align: center;
            margin-top: 4rem;
            padding: 2rem;
            background: #f7fafc;
            border-radius: 10px;
        }

        .footer p {
            color: #4a5568;
        }

        .footer a {
            color: #667eea;
            text-decoration: none;
            margin: 0 10px;
        }

        /* Expander customization */
        .streamlit-expanderHeader {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2d3748;
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .streamlit-expanderContent {
            background-color: white;
            padding: 1rem;
            color: #4a5568;
            line-height: 1.6;
            border-radius: 0 0 10px 10px;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    # Page configuration
    st.set_page_config(
        page_title="NTSC-BMP FAQ",
        page_icon="❓",
        layout="centered"
    )

    # Apply custom styles
    apply_custom_styles()

    # Header
    st.markdown("""
        <div class="faq-header">
            <h1>Frequently Asked Questions</h1>
            <p>Everything you need to know about NTSC-BMP</p>
        </div>
    """, unsafe_allow_html=True)

    # FAQ Content
    faqs = [
        {
            "question": "How does NTSC-BMP differ from existing compression algorithms?",
            "answer": "NTSC-BMP is unique because it doesn't treat all data equally. Instead, it assesses the importance of each data point, much like how our brains prioritize memories. By focusing on significant events and summarizing less important data, NTSC-BMP ensures that critical information is retained without unnecessary redundancy."
        },
        {
            "question": "Can NTSC-BMP be applied to real-time data streams?",
            "answer": "Absolutely! NTSC-BMP is designed to handle continuous, real-time data. Its multi-layered approach allows it to process incoming data on the fly, retaining immediate details when necessary and efficiently summarizing data over time based on its significance."
        },
        {
            "question": "How does the algorithm determine what data is significant?",
            "answer": "The algorithm uses statistical methods to analyze variance and detect anomalies within the data. Data points that show significant deviation or unusual patterns are flagged as important. This process can be customized with domain-specific thresholds to enhance accuracy for different applications."
        },
        {
            "question": "What are the computational requirements for NTSC-BMP?",
            "answer": "NTSC-BMP is designed to be computationally efficient. It uses lightweight statistical calculations that can be performed on standard hardware, and it's scalable to accommodate different data volumes. This makes it practical for a wide range of devices, including those with limited computational resources."
        },
        {
            "question": "How does NTSC-BMP handle data reconstruction?",
            "answer": "When reconstructing data, NTSC-BMP uses the significant data points it has retained and applies interpolation techniques to estimate the summarized or discarded data. While some level of detail is lost due to compression, the essential information remains intact, and the reconstructed data is sufficient for analysis and decision-making."
        },
        {
            "question": "Can NTSC-BMP integrate with existing systems?",
            "answer": "Yes, NTSC-BMP is designed to be flexible and can be integrated into current data pipelines and storage solutions. It can serve as a preprocessing step to optimize data before storage or transmission, enhancing the efficiency of existing systems without requiring a complete overhaul."
        }
    ]

    # Create FAQ section
    st.markdown('<div class="faq-section">', unsafe_allow_html=True)
    for faq in faqs:
        with st.expander(f"📌 {faq['question']}"):
            st.write(faq['answer'])
    st.markdown('</div>', unsafe_allow_html=True)

    # Navigation
    st.markdown("""
        <div class="navigation">
            <h3 style='color: #2d3748; margin-bottom: 1rem;'>Quick Navigation</h3>
            <a href="/" class="back-button">← Back to Main Page</a>
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="footer">
            <p>Still have questions? Feel free to reach out!</p>
            <div style='margin-top: 1rem;'>
                <a href="https://github.com/MuneebKhan11" target="_blank">GitHub</a>
                <a href="https://www.linkedin.com/in/khanmuneeb786/" target="_blank">LinkedIn</a>
            </div>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()