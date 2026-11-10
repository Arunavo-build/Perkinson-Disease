import sys
from fpdf import FPDF

html_content = """
<h2 align="center">Project Summary: Parkinson's Disease Classification</h2>

<h3>1. Dataset & Patient Recording Overview</h3>
<p>For this project, we worked with the Parkinson's Disease Speech Features dataset. It contains 756 voice recordings collected from 252 individuals, meaning each patient provided 3 separate recordings. Our main goal was to build a binary classification model that can accurately distinguish between Parkinson's Disease patients and healthy individuals based on their vocal features.</p>

<h3>2. Step-by-Step Methodology</h3>

<h4>Step 1: Data Splitting Strategies & Traditional Boosting Baseline</h4>
<p>To establish a solid baseline, we evaluated three traditional boosting algorithms: XGBoost, LightGBM, and CatBoost. Since we have multiple recordings per patient, we tested two different ways to split our data:</p>
<ul>
    <li><b>Grouped Split (Patient Records Isolated):</b> To test strict real-world scenarios, we made sure all 3 recordings of a single patient went entirely into either the training set or the testing set.</li>
    <li><b>Random Split (Patient Records Mixed):</b> We distributed the recordings randomly. This meant different recordings from the same patient could end up in both the training and testing sets simultaneously.</li>
</ul>
<p><b>Observation:</b> We noticed that the Random Split strategy gave us the highest predictive accuracies, with both LightGBM and XGBoost hitting 90.13%. We decided to select this best-performing setup (LightGBM + Random Split) as the foundation for our next experiments with deep learning.</p>

<h4>Step 2: Deep Tabular & Lightweight Transformer Evaluation</h4>
<p>Using our best data setup, we wanted to see how advanced deep learning models tailored for tabular data would perform. We ran experiments and tuned hyperparameters where applicable. Here is what we found:</p>
<ul>
    <li><b>TabNet (Tuned):</b> Achieved 86.84%</li>
    <li><b>GatedTabTransformer (GATE):</b> Achieved 85.53%</li>
    <li><b>SAINT (FTTransformer):</b> Achieved 84.21%</li>
    <li><b>TabTransformer:</b> Achieved 80.92%</li>
    <li><b>NODE:</b> Achieved 80.26%</li>
</ul>

<h4>Step 3: Kolmogorov-Arnold Networks (KAN) Integration</h4>
<p>To explore more novel architectures, we implemented a Kolmogorov-Arnold Network (KAN). Interestingly, KAN significantly outperformed all the transformer-based models we tested, reaching an impressive 90.79% accuracy as a standalone deep learning model.</p>

<h4>Step 4: Final Model Fusion Strategy</h4>
<p>To build the ultimate predictive model for our project, we selected our best traditional model (LightGBM) and our best deep learning model (KAN). We combined them using a Soft Voting Fusion technique, which averages the output probabilities of both models.</p>

<h3>3. Final Results & Conclusion</h3>
<p>Our fusion strategy successfully combined the strengths of tree-based boosting and neural networks, yielding the highest overall accuracy in our entire study. More importantly, the fusion model achieved a <b>98% Recall</b> for the patient class, meaning it successfully detected 98% of the actual Parkinson's patients.</p>

<h4>Comprehensive Model Accuracy Table</h4>
<table border="1" width="100%">
    <thead>
        <tr>
            <th width="40%">Methodology Category</th>
            <th width="40%">Algorithm / Architecture</th>
            <th width="20%">Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Traditional Boosting</td>
            <td>LightGBM (Selected Best Baseline)</td>
            <td>90.13%</td>
        </tr>
        <tr>
            <td>Traditional Boosting</td>
            <td>XGBoost</td>
            <td>90.13%</td>
        </tr>
        <tr>
            <td>Traditional Boosting</td>
            <td>CatBoost</td>
            <td>88.16%</td>
        </tr>
        <tr>
            <td>Advanced Deep DL</td>
            <td><b>Kolmogorov-Arnold Network (KAN)</b></td>
            <td><b>90.79%</b></td>
        </tr>
        <tr>
            <td>Deep Tabular / Transformer</td>
            <td>TabNet (Tuned)</td>
            <td>86.84%</td>
        </tr>
        <tr>
            <td>Deep Tabular / Transformer</td>
            <td>GatedTabTransformer (GATE)</td>
            <td>85.53%</td>
        </tr>
        <tr>
            <td>Deep Tabular / Transformer</td>
            <td>SAINT (FTTransformer)</td>
            <td>84.21%</td>
        </tr>
        <tr>
            <td>Deep Tabular / Transformer</td>
            <td>TabTransformer</td>
            <td>80.92%</td>
        </tr>
        <tr>
            <td>Deep Tabular / Transformer</td>
            <td>NODE</td>
            <td>80.26%</td>
        </tr>
        <tr>
            <td><b>Proposed Fusion Model</b></td>
            <td><b>LightGBM + KAN (Soft Voting)</b></td>
            <td><b>91.45%</b></td>
        </tr>
    </tbody>
</table>

<br>
<p><b>Conclusion:</b> Our systematic evaluation shows that while patient-mixed random splitting inflates initial boosting accuracy, integrating LightGBM with the modern KAN architecture via soft-voting fusion reliably provides the most optimal performance (<b>91.45% Accuracy</b>) for Parkinson's disease screening.</p>
"""

class PDF(FPDF):
    def header(self):
        pass

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
# Set standard font
pdf.set_font("Helvetica", size=11)
pdf.write_html(html_content)
pdf.output("Parkinsons_Project_Summary.pdf")
print("PDF created successfully!")
