import os
import re
import json
import shutil
import subprocess
import pandas as pd

WORKSPACE_DIR = r"d:\Perkinson disease"
ARTIFACT_DIR = r"C:\Users\Arunavo\.gemini\antigravity\brain\95ef30d1-213b-49db-a37e-a40ddf14f571"

TEX_PATH = os.path.join(WORKSPACE_DIR, "IEEE_Q1_Journal_Manuscript.tex")
HTML_PATH = os.path.join(WORKSPACE_DIR, "IEEE_Q1_Journal_Manuscript.html")
PDF_PATH = os.path.join(WORKSPACE_DIR, "IEEE_Q1_Journal_Manuscript.pdf")
PDF_ALIAS_PATH = os.path.join(WORKSPACE_DIR, "Q1_Journal_Manuscript.pdf")
PAPER_PDF_PATH = os.path.join(WORKSPACE_DIR, "Paper.pdf")
ARTIFACT_PDF_PATH = os.path.join(ARTIFACT_DIR, "IEEE_Q1_Journal_Manuscript.pdf")

SUPP_HTML_PATH = os.path.join(WORKSPACE_DIR, "IEEE_Q1_Supplementary_Material.html")
SUPP_PDF_PATH = os.path.join(WORKSPACE_DIR, "IEEE_Q1_Supplementary_Material.pdf")
ARTIFACT_SUPP_PDF_PATH = os.path.join(ARTIFACT_DIR, "IEEE_Q1_Supplementary_Material.pdf")

MD_PATH = os.path.join(WORKSPACE_DIR, "Q1_Journal_Manuscript.md")
ARTIFACT_MD_PATH = os.path.join(ARTIFACT_DIR, "Q1_Journal_Manuscript.md")

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_EXE):
    CHROME_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def build_references_list():
    refs_html_list = [
        '[1] W. Poewe, K. Seppi, C. M. Tanner, G. M. Halliday, P. Brundin, J. Volkmann, A.-E. Schrag, and A. E. Lang, &ldquo;Parkinson disease,&rdquo; <i>Nature Reviews Disease Primers</i>, vol. 3, Art. no. 17013, Mar. 2017, doi: 10.1038/nrdp.2017.13.',
        '[2] J. R. Duffy, <i>Motor Speech Disorders: Substrates, Differential Diagnosis, and Management</i>, 4th ed. St. Louis, MO, USA: Elsevier, 2019.',
        '[3] A. K. Ho, R. Iansek, C. Marigliani, J. L. Bradshaw, and S. Gates, &ldquo;Speech impairment in a large sample of patients with Parkinson&rsquo;s disease,&rdquo; <i>Behavioural Neurology</i>, vol. 11, no. 3, pp. 131&ndash;137, 1998, doi: 10.1155/1999/327643.',
        '[4] J. Hlavni&#269;ka, R. &#268;mejla, T. Tykalov&aacute;, K. &Scaron;onka, E. R&#367;&#382;i&#269;ka, and J. Rusz, &ldquo;Automated analysis of connected speech reveals early biomarkers of Parkinson&rsquo;s disease in patients with rapid eye movement sleep behaviour disorder,&rdquo; <i>Scientific Reports</i>, vol. 7, Art. no. 12, Jan. 2017, doi: 10.1038/s41598-017-00047-5.',
        '[5] J. Rusz, J. Hlavni&#269;ka, T. Tykalov&aacute;, M. Novotn&yacute;, P. Du&scaron;ek, K. &Scaron;onka, and E. R&#367;&#382;i&#269;ka, &ldquo;Smartphone allows capture of speech abnormalities associated with high risk of developing Parkinson&rsquo;s disease,&rdquo; <i>IEEE Trans. Neural Syst. Rehabil. Eng.</i>, vol. 26, no. 8, pp. 1495&ndash;1507, Aug. 2018, doi: 10.1109/TNSRE.2018.2851787.',
        '[6] L. O. Ramig, S. Sapir, S. Countryman, A. A. Pawlas, C. O&rsquo;Brien, M. Hoehn, and L. L. Thompson, &ldquo;Intensive voice treatment (LSVT) for patients with Parkinson&rsquo;s disease: A 2 year follow up,&rdquo; <i>J. Neurol. Neurosurg. Psychiatry</i>, vol. 71, no. 4, pp. 493&ndash;498, Oct. 2001, doi: 10.1136/jnnp.71.4.493.',
        '[7] M. A. Little, P. E. McSharry, S. J. Roberts, D. A. E. Costello, and I. M. Moroz, &ldquo;Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection,&rdquo; <i>BioMedical Engineering OnLine</i>, vol. 6, Art. no. 23, Jun. 2007, doi: 10.1186/1475-925X-6-23.',
        '[8] M. A. Little, P. E. McSharry, E. J. Hunter, J. Spielman, and L. O. Ramig, &ldquo;Suitability of dysphonia measurements for telemonitoring of Parkinson&rsquo;s disease,&rdquo; <i>IEEE Trans. Biomed. Eng.</i>, vol. 56, no. 4, pp. 1015&ndash;1022, Apr. 2009, doi: 10.1109/TBME.2008.2005954.',
        '[9] C. O. Sakar, G. Serbes, A. Gunduz, H. C. Tunc, H. Nizam, B. E. Sakar, M. Tutuncu, T. Aydin, M. E. Isenkul, and H. Apaydin, &ldquo;A comparative analysis of speech signal processing algorithms for Parkinson&rsquo;s disease classification and the use of the tunable Q-factor wavelet transform,&rdquo; <i>Applied Soft Computing</i>, vol. 74, pp. 255&ndash;263, Jan. 2019, doi: 10.1016/j.asoc.2018.10.022.',
        '[10] H. G&uuml;nd&uuml;z, &ldquo;Deep learning-based Parkinson&rsquo;s disease classification using vocal feature sets,&rdquo; <i>IEEE Access</i>, vol. 7, pp. 115540&ndash;115551, Aug. 2019, doi: 10.1109/ACCESS.2019.2936564.',
        '[11] R. Hossain, M. A. Kabir, A. I. G. Mowla, A. C. Roy, and R. K. Ghosh, &ldquo;BenSParX: A robust explainable machine learning framework for Parkinson&rsquo;s disease detection from Bengali conversational speech,&rdquo; <i>arXiv preprint arXiv:2505.12192</i>, May 2025, doi: 10.48550/arXiv.2505.12192.',
        '[12] I. Tougui, A. Jilbab, and J. El Mhamdi, &ldquo;Impact of the choice of cross-validation techniques on the results of machine learning-based diagnostic applications,&rdquo; <i>Healthcare Informatics Research</i>, vol. 27, no. 3, pp. 189&ndash;199, Jul. 2021, doi: 10.4258/hir.2021.27.3.189.',
        '[13] S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, and K. P. Kording, &ldquo;The need to approximate the use-case in clinical machine learning,&rdquo; <i>GigaScience</i>, vol. 6, no. 5, Art. no. gix019, May 2017, doi: 10.1093/gigascience/gix019.',
        '[14] L. Moro-Velazquez, J. A. Gomez-Garcia, J. D. Arias-Londo&ntilde;o, N. Dehak, and J. I. Godino-Llorente, &ldquo;Advances in Parkinson&rsquo;s disease detection and assessment using voice and speech: A review of the articulatory and phonatory aspects,&rdquo; <i>Biomed. Signal Process. Control</i>, vol. 66, Art. no. 102418, Apr. 2021, doi: 10.1016/j.bspc.2021.102418.',
        '[15] A. Favaro, Y.-T. Tsai, A. Butala, T. Thebaud, J. Villalba, N. Dehak, and L. Moro-Vel&aacute;zquez, &ldquo;Interpretable speech features vs. DNN embeddings: What to use in the automatic assessment of Parkinson&rsquo;s disease in multi-lingual scenarios,&rdquo; <i>Comput. Biol. Med.</i>, vol. 166, Art. no. 107559, Nov. 2023, doi: 10.1016/j.compbiomed.2023.107559.',
        '[16] T. Chen and C. Guestrin, &ldquo;XGBoost: A scalable tree boosting system,&rdquo; in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining (KDD)</i>, 2016, pp. 785&ndash;794, doi: 10.1145/2939672.2939785.',
        '[17] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, &ldquo;LightGBM: A highly efficient gradient boosting decision tree,&rdquo; in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017, pp. 3146&ndash;3154.',
        '[18] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, &ldquo;CatBoost: Unbiased boosting with categorical features,&rdquo; in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 31, 2018, pp. 6639&ndash;6649.',
        '[19] S. &Ouml;. Arik and T. Pfister, &ldquo;TabNet: Attentive interpretable tabular learning,&rdquo; in <i>Proc. AAAI Conf. Artif. Intell.</i>, vol. 35, no. 8, May 2021, pp. 6679&ndash;6687, doi: 10.1609/aaai.v35i8.16826.',
        '[20] B. E. Sakar, M. E. Isenkul, C. O. Sakar, A. Sertbas, F. Gurgen, S. Delil, H. Apaydin, and O. Kursun, &ldquo;Collection and analysis of a Parkinson speech dataset with multiple types of sound recordings,&rdquo; <i>IEEE J. Biomed. Health Inform.</i>, vol. 17, no. 4, pp. 828&ndash;834, Jul. 2013, doi: 10.1109/JBHI.2013.2245674.',
        '[21] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, &ldquo;Revisiting deep learning models for tabular data,&rdquo; in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 34, 2021, pp. 18932&ndash;18943.',
        '[22] G. Somepalli, A. Schwarzschild, M. Goldblum, C. B. Bruss, and T. Goldstein, &ldquo;SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training,&rdquo; <i>arXiv preprint arXiv:2106.01342</i>, Jun. 2021, doi: 10.48550/arXiv.2106.01342.',
        '[23] S. Popov, S. Morozov, and A. Babenko, &ldquo;Neural oblivious decision ensembles for deep learning on tabular data,&rdquo; in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2020.',
        '[24] M. Joseph and H. Raj, &ldquo;GANDALF: Gated adaptive network for deep automated learning of features,&rdquo; <i>arXiv preprint arXiv:2207.08548</i>, Jul. 2022, doi: 10.48550/arXiv.2207.08548.',
        '[25] L. Grinsztajn, E. Oyallon, and G. Varoquaux, &ldquo;Why do tree-based models still outperform deep learning on typical tabular data?&rdquo; in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 35, 2022, pp. 507&ndash;520.',
        '[26] E. Chaibub Neto <i>et al.</i>, &ldquo;Detecting the impact of subject characteristics on machine learning-based diagnostic applications,&rdquo; <i>npj Digital Medicine</i>, vol. 2, Art. no. 99, Oct. 2019, doi: 10.1038/s41746-019-0178-x.',
        '[27] H. G&uuml;nd&uuml;z, &ldquo;An efficient dimensionality reduction method using filter-based feature selection and variational autoencoders on Parkinson&rsquo;s disease classification,&rdquo; <i>Biomed. Signal Process. Control</i>, vol. 66, Art. no. 102452, Apr. 2021, doi: 10.1016/j.bspc.2021.102452.',
        '[28] M. Hire&scaron;, M. Gazda, P. Drot&aacute;r, N. D. Pah, M. A. Motin, and D. K. Kumar, &ldquo;Convolutional neural network ensemble for Parkinson&rsquo;s disease detection from voice recordings,&rdquo; <i>Comput. Biol. Med.</i>, vol. 141, Art. no. 105021, Feb. 2022, doi: 10.1016/j.compbiomed.2021.105021.',
        '[29] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Solja&#269;i&#263;, T. Y. Hou, and M. Tegmark, &ldquo;KAN: Kolmogorov-Arnold Networks,&rdquo; in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2025, doi: 10.48550/arXiv.2404.19756.',
        '[30] S. M. Lundberg and S.-I. Lee, &ldquo;A unified approach to interpreting model predictions,&rdquo; in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017, pp. 4765&ndash;4774.',
        '[31] S. M. Lundberg <i>et al.</i>, &ldquo;From local explanations to global understanding with explainable AI for trees,&rdquo; <i>Nature Machine Intelligence</i>, vol. 2, no. 1, pp. 56&ndash;67, Jan. 2020, doi: 10.1038/s42256-019-0138-9.',
        '[32] B. Sun, J. Feng, and K. Saenko, &ldquo;Return of frustratingly easy domain adaptation,&rdquo; in <i>Proc. 30th AAAI Conf. Artif. Intell.</i>, vol. 30, no. 1, Feb. 2016, pp. 2058&ndash;2065, doi: 10.1609/aaai.v30i1.10306.',
        '[33] A. Favaro <i>et al.</i>, &ldquo;Multilingual evaluation of interpretable biomarkers to represent language and speech patterns in Parkinson&rsquo;s disease,&rdquo; <i>Frontiers in Neurology</i>, vol. 14, Art. no. 1142642, Mar. 2023, doi: 10.3389/fneur.2023.1142642.',
        '[34] G. Solana-Lavalle and R. Rosas-Romero, &ldquo;Analysis of voice as an assisting tool for detection of Parkinson&rsquo;s disease and its subsequent clinical interpretation,&rdquo; <i>Biomed. Signal Process. Control</i>, vol. 66, Art. no. 102415, Apr. 2021, doi: 10.1016/j.bspc.2021.102415.',
        '[35] N. A. Obuchowski, &ldquo;On the comparison of correlated proportions for clustered data,&rdquo; <i>Statistics in Medicine</i>, vol. 17, no. 13, pp. 1495&ndash;1507, Jul. 1998, doi: 10.1002/(SICI)1097-0258(19980715)17:13&lt;1495::AID-SIM863&gt;3.0.CO;2-I.'
    ]
    return refs_html_list


def convert_inline_code_and_math(html_str):
    # Convert backtick `code` to <code>code</code>
    html_str = re.sub(r"`([^`]+)`", r"<code>\1</code>", html_str)
    return html_str


def generate_main_html(fig_dir, refs):
    ref_items_html = [f'<div class="ref-item">{r}</div>' for r in refs]
    refs_html = "\n".join(ref_items_html)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Patient-Grouped Parkinson's Disease Screening Across Languages via Tree-Spline Fusion</title>
<style>
  @page {
    size: letter portrait;
    margin: 0.40in 0.42in 0.40in 0.42in;
  }
  * {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }
  body {
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 8.6pt;
    line-height: 1.20;
    color: #000000;
    margin: 0;
    padding: 0;
    background: #ffffff;
  }
  code {
    font-family: "Courier New", Courier, monospace;
    font-size: 7.9pt;
  }
  .ieee-running-header {
    font-size: 6.9pt;
    letter-spacing: 0.2px;
    text-transform: uppercase;
    border-bottom: 0.6pt solid #000;
    padding-bottom: 2px;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
    color: #111;
  }
  .ieee-title-block {
    text-align: center;
    margin-bottom: 5px;
  }
  .ieee-title {
    font-size: 15.2pt;
    font-weight: normal;
    line-height: 1.13;
    margin: 0 0 4px 0;
  }
  .ieee-authors {
    font-size: 9.6pt;
    margin-bottom: 2px;
  }
  .ieee-affil {
    font-size: 7.9pt;
    color: #222;
    margin-bottom: 2px;
  }
  .two-col {
    column-count: 2;
    column-gap: 0.18in;
    text-align: justify;
    hyphens: auto;
    margin-bottom: 3px;
  }
  .full-width-float {
    width: 100%;
    margin: 3.5px 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .ieee-abstract {
    font-size: 8.1pt;
    font-weight: bold;
    line-height: 1.18;
    margin-bottom: 4px;
    text-align: justify;
    text-indent: 0;
  }
  .ieee-keywords {
    font-size: 8.0pt;
    font-weight: bold;
    line-height: 1.17;
    margin-bottom: 5px;
    text-align: justify;
    text-indent: 0;
  }
  .dropcap {
    float: left;
    font-size: 21.5pt;
    line-height: 0.82;
    padding-top: 2px;
    padding-right: 3px;
    font-weight: normal;
  }
  .sc {
    font-variant: small-caps;
  }
  p {
    margin: 0 0 3.2px 0;
    text-indent: 0.9em;
  }
  p.no-indent {
    text-indent: 0;
  }
  h2.ieee-sec {
    font-size: 9.1pt;
    font-weight: normal;
    font-variant: small-caps;
    text-align: center;
    margin: 5px 0 2.5px 0;
    break-after: avoid;
    page-break-after: avoid;
  }
  h3.ieee-subsec {
    font-size: 8.55pt;
    font-weight: normal;
    font-style: italic;
    text-align: left;
    margin: 4px 0 2px 0;
    break-after: avoid;
    page-break-after: avoid;
  }
  .eq-block {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 2.5px 0;
    padding: 1px 4px;
    font-size: 8.3pt;
    break-inside: avoid;
  }
  .eq-formula {
    text-align: center;
    flex-grow: 1;
  }
  .eq-num {
    margin-left: 6px;
    white-space: nowrap;
  }
  .ieee-table-wrap {
    width: 100%;
    margin: 3.5px 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .ieee-table-title {
    text-align: center;
    font-size: 7.3pt;
    text-transform: uppercase;
    letter-spacing: 0.25px;
    margin-bottom: 2px;
    line-height: 1.14;
  }
  table.ieee-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 7.05pt;
    line-height: 1.12;
    border-top: 1.1pt solid #000;
    border-bottom: 1.1pt solid #000;
  }
  table.ieee-table thead th {
    border-top: 0.5pt solid #000;
    border-bottom: 0.6pt solid #000;
    padding: 1.8px 3px;
    font-weight: bold;
    text-align: center;
    vertical-align: bottom;
  }
  table.ieee-table tbody td {
    padding: 1.4px 3px;
    text-align: center;
    vertical-align: middle;
  }
  table.ieee-table tbody td.l, table.ieee-table thead th.l {
    text-align: left;
  }
  table.ieee-table tbody tr.sec-divider td {
    border-top: 0.5pt solid #444;
  }
  table.ieee-table tbody tr.highlight-row td {
    font-weight: bold;
    background-color: #f8fafc;
  }
  .table-note {
    font-size: 6.7pt;
    text-align: justify;
    margin-top: 1.5px;
    line-height: 1.11;
  }
  .ieee-fig {
    width: 100%;
    margin: 2.5px 0;
    text-align: center;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .ieee-fig img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
  }
  .ieee-fig-cap {
    font-size: 7.25pt;
    line-height: 1.13;
    text-align: justify;
    margin-top: 2px;
  }
  .ref-list {
    font-size: 6.9pt;
    line-height: 1.11;
  }
  .ref-item {
    margin-bottom: 1.2px;
    padding-left: 1.5em;
    text-indent: -1.5em;
    text-align: justify;
  }
</style>
</head>
<body>

<div class="ieee-running-header">
  <span>IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 30, 2026</span>
  <span>PATIENT-GROUPED PARKINSON'S SPEECH SCREENING ACROSS LANGUAGES</span>
</div>

<div class="ieee-title-block">
  <h1 class="ieee-title">Patient-Grouped Parkinson's Disease Screening Across Languages via LightGBM and Kolmogorov&ndash;Arnold Spline Ensembles</h1>
  <div class="ieee-authors">Arunavo, <i>Member, IEEE</i></div>
  <div class="ieee-affil">Department of Computer Science and Biomedical Engineering &bull; Code &amp; Data: https://github.com/Arunavo-build/Perkinson-Disease</div>
</div>

<div class="two-col">
  <div class="ieee-abstract">
    <i>Abstract</i>&mdash;Voice recordings offer a practical, non-invasive way to screen for Parkinson's disease (PD), yet many published studies split multiple recordings from the same speaker across training and test sets. This record-wise evaluation allows classifiers to memorize speaker identity rather than vocal pathology, inflating reported accuracy by 6.6% to 16.8% on standard benchmarks. Furthermore, acoustic features extracted from sustained vowel phonation and spontaneous conversation differ in both dimensionality and physical meaning, complicating deployment across languages. In this study, we evaluate a patient-grouped classification framework across three cohorts: Turkish sustained vowels (TR, 756 recordings from 252 subjects), Bengali spontaneous speech (BN, 900 recordings from 120 subjects), and English sustained vowels (OX, 195 recordings from 32 subjects). To combine sharp decision boundaries on perturbation thresholds with smooth modeling of continuous spectral trajectories, we average probability estimates from gradient-boosted decision trees (LightGBM) and cubic B-spline Kolmogorov&ndash;Arnold Networks (KAN). Across five random seeds under strict five-fold patient-wise cross-validation, the combined model achieves recording-level accuracies of 85.08% on TR, 93.49% on BN, and 76.82% on OX, rising to 86.27%, 100.00%, and 81.25% when aggregating repeated recordings per subject. Calibrating decision thresholds inside each training fold increases specificity on imbalanced cohorts from roughly 51% to over 71%. Finally, few-shot cross-lingual experiments show that while training directly on abundant target-language data remains superior, transferring a source model aligned via Correlation Alignment (CORAL) improves screening accuracy by 5.5% to 13.0% when only 5% to 10% of target patients are available.
  </div>

  <div class="ieee-keywords">
    <i>Index Terms</i>&mdash;Acoustic biomarkers, cross-lingual domain adaptation, gradient-boosted decision trees, Kolmogorov&ndash;Arnold Networks (KAN), Parkinson's disease (PD), patient-wise cross-validation.
  </div>

  <h2 class="ieee-sec">I. Introduction</h2>
  <p class="no-indent"><span class="dropcap">P</span><span class="sc">arkinson's</span> disease (PD) is a progressive neurodegenerative disorder driven by the loss of dopaminergic neurons in the substantia nigra pars compacta [1], [2]. Because motor pathways governing laryngeal and articulatory musculature are affected early in the disease course, up to 90% of patients develop hypokinetic dysarthria&mdash;characterized by reduced vocal intensity, monopitch, breathy phonation, and irregular articulatory transitions&mdash;often several years before cardinal limb tremor or bradykinesia prompts a clinical diagnosis [3], [4]. Acoustic analysis of voice recordings therefore provides an inexpensive, non-invasive screening tool suitable for remote telemedicine [5], [6].</p>

  <p>Despite encouraging results in the literature, three methodological issues limit the clinical reliability of many existing voice-based PD classifiers:</p>

  <p><i>1) Record-Wise Splitting and Speaker Identity Leakage:</i> Standard PD speech datasets record multiple utterances from each participant (three vowels per subject in the Turkish UCI corpus [9], six in the English Oxford corpus [7], and seven to eight conversational segments in the Bengali BenSParX corpus [11]). Studies that apply standard record-wise <i>k</i>-fold cross-validation place recordings from the same individual in both training and test folds [12], [13]. High-capacity models then recognize speaker-specific vocal tract anatomy and microphone background rather than disease-related dysphonia. When evaluated on genuinely unseen patients via subject-wise splitting, as properly done by Sakar <i>et al.</i> [9] and G&uuml;nd&uuml;z [10], accuracies drop substantially compared to leaked record-wise reports.</p>

  <p><i>2) Class Imbalance and Low Specificity at Default Thresholds:</i> Clinical PD cohorts are frequently skewed toward positive cases (roughly 75% PD in the Turkish and Oxford datasets). Training a classifier under standard cross-entropy loss and applying an unadjusted decision threshold (<i>&tau;</i> = 0.50) biases predictions toward the majority class, yielding high sensitivity (&gt;95%) but poor specificity (~50%), which would flag roughly half of healthy screening participants as false positives.</p>

  <p><i>3) Task Mismatch in Cross-Lingual Transfer:</i> Most public voice datasets are collected in a single language and recording protocol. Transferring an acoustic model between sustained vowel phonation (<code>/a/</code> in Turkish or English) and spontaneous conversational speech (Bengali) is non-trivial: static Mel-frequency cepstral coefficients (MFCCs) and dynamic spectral derivatives capture different articulatory mechanics across stationary vowels and running speech [14], [15].</p>

  <p>To address these practical challenges, we present a leak-free, multi-seed evaluation of tree-based and spline-based classifiers across Turkish, Bengali, and English PD speech cohorts. Our specific contributions are as follows:</p>
  <p>&bull; We pair gradient-boosted decision trees (LightGBM [17]) with cubic B-spline Kolmogorov&ndash;Arnold Networks (KAN [29]) and compare this combination against standalone models, multi-layer perceptrons (LightGBM+MLP), support vector machines (LightGBM+SVM), logistic stacking, and ten tabular deep learning baselines across five random seeds under strict patient-grouped cross-validation.</p>
  <p>&bull; We evaluate both recording-level predictions and subject-level aggregation across replicate recordings per speaker, and apply inner-fold Youden's <i>J</i> threshold calibration (<i>&tau;</i><sup>*</sup>) to recover clinical specificity on imbalanced cohorts without test leakage.</p>
  <p>&bull; We perform patient-clustered statistical testing using Obuchowski's clustered McNemar test [35] and 25-fold multi-seed Wilcoxon signed-rank tests, accounting for non-independence among repeated recordings per patient.</p>
  <p>&bull; We analyze cross-lingual biomarker directionality&mdash;explaining why static energy tremor and formant envelopes maintain consistent polarity across languages whereas cepstral velocity (<code>Delta_MFCC_2</code>) flips sign between sustained vowels and conversation&mdash;and map out few-shot target learning curves (5% to 50% labeled patients) with and without Correlation Alignment (CORAL [32]).</p>
</div>

<div class="full-width-float">
  <div class="ieee-fig">
    <img src="{fig_dir}/Fig1_System_Architecture.png" style="max-height: 2.02in;" alt="Fig. 1 Graphical System Overview">
    <div class="ieee-fig-cap"><b>Fig. 1.</b> Graphical overview of the patient-grouped Parkinson's speech screening framework. <b>(a)</b> Normalized acoustic waveforms comparing periodic glottal cycles in Healthy Controls (HC) against pitch jitter (&Delta;<i>T<sub>i</sub></i>) and amplitude shimmer (&Delta;<i>A<sub>i</sub></i>) in Parkinson's Disease (PD) across Turkish (TR, sustained vowel /a/), Bengali (BN, spontaneous speech), and English Oxford (OX, sustained vowel /a/) cohorts. <b>(b)</b> Strict patient-wise five-fold GroupKFold split (<i>S</i><sub>train</sub> &cap; <i>S</i><sub>test</sub> = &empty;) isolating all 3&ndash;8 recordings of each speaker within a single fold. <b>(c)</b> Comparison of LightGBM orthogonal step splits <i>f</i><sub>tree</sub>(<i>z</i>) and smooth KAN cubic B-spline edge activations <i>&phi;<sub>i</sub></i>(<i>z</i>). <b>(d)</b> Out-of-fold probability fusion, inner-fold Youden's <i>&tau;</i><sup>*</sup> threshold calibration to reduce false positives on 3:1 imbalanced cohorts, and subject-level voting.</div>
  </div>
</div>

<div class="two-col">
  <h2 class="ieee-sec">II. Related Work</h2>
  <p class="no-indent">Early telemonitoring studies by Little <i>et al.</i> [7] and Tsanas <i>et al.</i> [8] introduced nonlinear dysphonia measures such as pitch period entropy (PPE) and recurrence period density entropy (RPDE) on the 32-subject Oxford corpus. In 2019, Sakar <i>et al.</i> [9] released the 252-subject Turkish UCI dataset combining baseline perturbation features, MFCCs, and tunable Q-factor wavelet transform (TQWT) coefficients, reporting 86.0% accuracy using support vector machines under leave-one-subject-out (LOSO) validation. G&uuml;nd&uuml;z [10] subsequently applied deep convolutional networks with subject-level leave-one-person-out (LOPO) pooling on the same Turkish cohort, achieving 86.9% subject-level accuracy. For Bengali, Hossain <i>et al.</i> [11] curated the 120-subject BenSParX conversational dataset and reported up to 93.78% accuracy using gradient boosting and feature selection. By contrast, several recent deep learning studies on the Turkish and Oxford datasets have reported 92%&ndash;98% accuracy by inadvertently using record-wise random splits [12], [13]; as quantified by Rusz <i>et al.</i> [15] and Moro-Velazquez <i>et al.</i> [14], such splits leak speaker identity across folds.</p>

  <p>In tabular machine learning, gradient-boosted decision trees (XGBoost [16], LightGBM [17], CatBoost [18]) consistently outperform deep neural architectures (TabNet [19], TabTransformer [20], FT-Transformer [21], SAINT [22], NODE [23], GATE [24]) on medium-sized clinical datasets because orthogonal axis-aligned splits naturally handle heterogeneous feature scales and sharp clinical cutoffs [25], [26]. Recently, Liu <i>et al.</i> [29] introduced Kolmogorov&ndash;Arnold Networks (KAN), which replace fixed node activations with learnable univariate cubic B-splines on network edges. While standalone KANs can overfit noisy high-dimensional features, their smooth piecewise-polynomial curves model continuous acoustic gradations that step-function trees approximate coarsely. We investigate whether combining LightGBM with KAN yields better probability calibration and interpretability than pairing LightGBM with standard MLPs, SVMs, or linear meta-learners.</p>

  <h2 class="ieee-sec">III. Materials and Methods</h2>
  <h3 class="ieee-subsec">A. Multilingual Speech Corpora and Leak-Free Partitioning</h3>
  <p class="no-indent">We evaluate three public dysarthria datasets spanning Turkic, Indo-Aryan, and Germanic language families (summarized in Table I): <b>1) Turkish UCI Corpus (TR)</b> [9], comprising <i>N</i> = 756 sustained <code>/a/</code> vowel recordings (3 replicates per subject) from <i>S</i> = 252 speakers (188 PD, 64 HC; 74.6% PD) with <i>D</i> = 753 acoustic features; <b>2) Bengali BenSParX Corpus (BN)</b> [11], comprising <i>N</i> = 900 spontaneous speech segments (7&ndash;8 segments per subject) from <i>S</i> = 120 speakers (60 PD, 60 HC; 50.0% PD) with <i>D</i> = 71 features; and <b>3) English Oxford Corpus (OX)</b> [7], comprising <i>N</i> = 195 sustained <code>/a/</code> phonations (6 replicates per subject) from <i>S</i> = 32 speakers (23 PD, 8 HC; 75.4% PD) with <i>D</i> = 22 features. Standard mathematical definitions for jitter, shimmer, harmonics-to-noise ratio (HNR), MFCCs, and TQWT wavelet energy/entropy are provided in Supplementary Section S1.</p>

  <p>To prevent speaker identity leakage, all cross-validation splits are constructed at the subject level using five-fold patient-grouped cross-validation (<code>GroupKFold</code>). Let <i>s<sub>i</sub></i> &isin; {1, &hellip;, <i>S</i>} denote the speaker identity of recording <i>i</i>. The training and test subject sets in every fold <i>k</i> &isin; {1, &hellip;, 5} satisfy</p>
  <div class="eq-block">
    <div class="eq-formula"><i>S</i><sub>train</sub><sup>(<i>k</i>)</sup> &cap; <i>S</i><sub>test</sub><sup>(<i>k</i>)</sup> = &empty;, &emsp; &forall;<i>i</i> &isin; <i>D</i><sub>train</sub><sup>(<i>k</i>)</sup>, &forall;<i>j</i> &isin; <i>D</i><sub>test</sub><sup>(<i>k</i>)</sup> : <i>s<sub>i</sub></i> &ne; <i>s<sub>j</sub></i>.</div>
    <div class="eq-num">(1)</div>
  </div>
  <p class="no-indent">Feature standardization (<i>z</i>-score normalization) and any feature selection or threshold tuning are fitted strictly inside <i>D</i><sub>train</sub><sup>(<i>k</i>)</sup>. To quantify initialization and partitioning variance, all experiments are repeated across five independent random seeds (<code>{42, 100, 2024, 777, 999}</code>). Because each patient contributes <i>m<sub>i</sub></i> &isin; [3, 8] recordings, we report both <i>recording-level</i> metrics and <i>subject-level voting</i> metrics (obtained by averaging the out-of-fold predicted probabilities across all recordings belonging to subject <i>s</i>).</p>

  <h3 class="ieee-subsec">B. LightGBM and Spline-Parameterized KAN</h3>
  <p class="no-indent">Figure 1 illustrates our classification pipeline. The first branch is a leaf-wise gradient-boosted decision tree ensemble (LightGBM [17]) with <i>M</i> = 100 trees, maximum leaves <i>L</i><sub>max</sub> = 31, feature fraction 0.85, and learning rate <i>&eta;</i> &isin; {0.05, 0.10}. Because each tree partitions the feature space via axis-aligned indicator steps I(<i>x<sub>j</sub></i> &le; <i>&theta;</i>), LightGBM effectively isolates sharp perturbation thresholds and multi-way interactions among TQWT wavelet bands and glottal pulse features.</p>

  <p>The second branch is a Kolmogorov&ndash;Arnold Network (KAN [29]) with architecture [<i>D</i><sub>in</sub>, 8, 2]. Unlike a standard MLP with fixed ReLU node activations, each edge (<i>p</i>, <i>q</i>) in KAN computes a learnable univariate transformation <i>&phi;<sub>p,q</sub></i>(<i>z</i>) combining a SiLU base function with a cubic B-spline (<i>k</i> = 3) over <i>G</i> = 5 grid intervals:</p>
  <div class="eq-block">
    <div class="eq-formula"><i>&phi;<sub>p,q</sub></i>(<i>z</i>) = <i>w<sub>b</sub></i> [<i>z</i> / (1 + <i>e</i><sup>&minus;<i>z</i></sup>)] + <i>w<sub>s</sub></i> &Sigma;<sub><i>m</i>=1</sub><sup><i>G</i>+<i>k</i></sup> <i>c<sub>p,q,m</sub></i> <i>B<sub>m,k</sub></i>(<i>z</i>),</div>
    <div class="eq-num">(2)</div>
  </div>
  <p class="no-indent">where the B-spline basis functions <i>B<sub>m,k</sub></i>(<i>z</i>) are evaluated via the Cox&ndash;de Boor recursion (Supplementary Eq. S6) and trained with Adam (lr = 10<sup>&minus;2</sup>, 60 epochs) under <i>L</i><sub>1</sub> and entropy regularization. Full hyperparameter grids and tuning protocols for LightGBM, KAN, and all ten deep tabular baselines are detailed in Supplementary Section S2.</p>

  <h3 class="ieee-subsec">C. Out-of-Fold Probability Fusion and Threshold Calibration</h3>
  <p class="no-indent">For a test recording <b>x</b>, the combined posterior probability is a convex average of the tree and spline branches, <i>P</i><sub>fuse</sub>(<i>y</i> = 1 | <b>x</b>) = <i>&alpha; P</i><sub>LGBM</sub>(<i>y</i> = 1 | <b>x</b>) + (1 &minus; <i>&alpha;</i>) <i>P</i><sub>KAN</sub>(<i>y</i> = 1 | <b>x</b>), where the mixing weight <i>&alpha;</i> &isin; [0, 1] is selected using out-of-fold predictions within the training split. On imbalanced cohorts such as TR (74.6% PD) and OX (75.4% PD), applying an unadjusted threshold <i>&tau;</i> = 0.50 maximizes raw accuracy by over-predicting the majority PD class, resulting in ~50% specificity. To provide a balanced clinical operating point without test leakage, we compute the receiver operating characteristic (ROC) curve strictly on the training fold out-of-fold probabilities and select the threshold <i>&tau;</i><sup>*</sup> maximizing Youden's <i>J</i> index:</p>
  <div class="eq-block">
    <div class="eq-formula"><i>&tau;</i><sup>*</sup> = arg max<sub><i>&tau;</i> &isin; [0.2, 0.9]</sub> (Sensitivity<sub>train</sub>(<i>&tau;</i>) + Specificity<sub>train</sub>(<i>&tau;</i>) &minus; 1).</div>
    <div class="eq-num">(3)</div>
  </div>

  <h3 class="ieee-subsec">D. Patient-Clustered Statistical Tests &amp; Cross-Lingual Alignment</h3>
  <p class="no-indent">Standard McNemar tests and Wilson score intervals assume that all <i>N</i> recordings are statistically independent. Because each patient contributes <i>m<sub>i</sub></i> correlated recordings, treating recordings as independent underestimates variance and produces overly optimistic <i>p</i>-values. Let <i>d<sub>ij</sub></i> = I(<i>&#375;<sub>ij</sub></i><sup>(<i>A</i>)</sup> = <i>y<sub>ij</sub></i>) &minus; I(<i>&#375;<sub>ij</sub></i><sup>(<i>B</i>)</sup> = <i>y<sub>ij</sub></i>) &isin; {&minus;1, 0, 1} denote the paired accuracy difference between models <i>A</i> and <i>B</i> on recording <i>j</i> of subject <i>i</i>, and let <i>D<sub>i</sub></i> = &Sigma;<sub><i>j</i>=1</sub><sup><i>m<sub>i</sub></i></sup> <i>d<sub>ij</sub></i> be the subject-level cluster sum across <i>S</i> independent patients. Following Obuchowski [35], the cluster-adjusted paired chi-square statistic is</p>
  <div class="eq-block">
    <div class="eq-formula"><i>&chi;</i><sup>2</sup><sub>clus</sub> = (&Sigma;<sub><i>i</i>=1</sub><sup><i>S</i></sup> <i>D<sub>i</sub></i>)<sup>2</sup> / [ (<i>S</i> / (<i>S</i> &minus; 1)) &Sigma;<sub><i>i</i>=1</sub><sup><i>S</i></sup> (<i>D<sub>i</sub></i> &minus; <i>D&#772;</i>)<sup>2</sup> ] ~ <i>&chi;</i><sup>2</sup><sub>1</sub>,</div>
    <div class="eq-num">(4)</div>
  </div>
  <p class="no-indent">which we report alongside exact binomial McNemar tests on the <i>S</i> aggregated subject-level predictions and 25-fold Wilcoxon signed-rank tests across all 5 seeds &times; 5 folds.</p>

  <p>For cross-lingual transfer between TR (<i>D</i> = 753) and BN (<i>D</i> = 71), we identify <i>D</i><sub>shared</sub> = 27 homologous acoustic features (and 13 shared dysphonia features for OX) covering glottal pulse periodicity, jitter (<code>locPctJitter</code>, <code>rapJitter</code>, <code>ppq5Jitter</code>), shimmer (<code>locShimmer</code>, <code>apq3Shimmer</code>, <code>apq5Shimmer</code>), harmonicity (<code>HNR</code>, autocorrelation), formants (<code>F1</code>&ndash;<code>F3</code>), static <code>MFCC_1</code>&ndash;<code>MFCC_6</code>, cepstral velocities, and log-energy acceleration (complete mapping and extraction differences in Supplementary Tables S2&ndash;S3). Because MFCCs and perturbation measures from sustained vowels (<code>/a/</code>) and conversational speech exhibit substantial covariate shift, we evaluate unsupervised second-order Correlation Alignment (CORAL [32]):</p>
  <div class="eq-block">
    <div class="eq-formula"><b>X</b><sub><i>S</i></sub><sup>aligned</sup> = (<b>X</b><sub><i>S</i></sub> &minus; <b><i>&mu;</i></b><sub><i>S</i></sub>)(<b>C</b><sub><i>S</i></sub> + <i>&lambda;</i><b>I</b>)<sup>&minus;1/2</sup>(<b>C</b><sub><i>T</i></sub> + <i>&lambda;</i><b>I</b>)<sup>1/2</sup>,</div>
    <div class="eq-num">(5)</div>
  </div>
  <p class="no-indent">where <b>C</b><sub><i>S</i></sub>, <b>C</b><sub><i>T</i></sub> are the source and unlabeled target covariance matrices (<i>&lambda;</i> = 10<sup>&minus;3</sup>), and compare target-only training against source+target transfer across 0% to 50% labeled target patient budgets.</p>
</div>

<div class="full-width-float">
  <div class="ieee-table-wrap">
    <div class="ieee-table-title"><b>Table I</b><br>Summary of Multilingual Parkinson's Speech Corpora and Leak-Free Patient-Wise Benchmarks</div>
    <table class="ieee-table">
      <thead>
        <tr>
          <th class="l">Corpus (Code) &amp; Language</th>
          <th>Speech Task</th>
          <th>Recordings (<i>N</i>)</th>
          <th>Subjects (<i>S</i>)</th>
          <th>Class Ratio (PD : HC)</th>
          <th>Features (<i>D</i>)</th>
          <th class="l">Prior Leak-Free Benchmark (Protocol)</th>
          <th>Ours: Recording Acc</th>
          <th>Ours: Subject Acc</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="l"><b>Turkish UCI (TR)</b> [9] (Turkish)</td>
          <td>Sustained Vowel /a/</td>
          <td>756 (3/subj)</td>
          <td>252</td>
          <td>188 : 64 (74.6% PD)</td>
          <td>753</td>
          <td class="l">Sakar <i>et al.</i> [9]: 86.0% (LOSO); G&uuml;nd&uuml;z [10]: 86.9% (LOPO Subj.)</td>
          <td>85.08 &plusmn; 0.49%</td>
          <td><b>86.27 &plusmn; 0.74%</b></td>
        </tr>
        <tr>
          <td class="l"><b>BenSParX (BN)</b> [11] (Bengali)</td>
          <td>Spontaneous Speech</td>
          <td>900 (7&ndash;8/subj)</td>
          <td>120</td>
          <td>60 : 60 (50.0% PD)</td>
          <td>71</td>
          <td class="l">Hossain <i>et al.</i> [11]: 93.78% (Single Split); LGBM+MLP: 94.16%</td>
          <td>93.49 &plusmn; 0.36%</td>
          <td><b>100.00 &plusmn; 0.00%</b></td>
        </tr>
        <tr>
          <td class="l"><b>Oxford UCI (OX)</b> [7] (English)</td>
          <td>Sustained Vowel /a/</td>
          <td>195 (6/subj)</td>
          <td>32</td>
          <td>23 : 8 (75.4% PD)</td>
          <td>22</td>
          <td class="l">Little <i>et al.</i> [7]: 91.4% (Record-Wise Leaked); 75.6% (5-Fold Subj.)</td>
          <td>76.82 &plusmn; 3.33%</td>
          <td><b>81.25 &plusmn; 3.95%</b></td>
        </tr>
      </tbody>
    </table>
    <div class="table-note">Abbreviations: TR = Turkish UCI Corpus; BN = Bengali BenSParX Corpus; OX = English Oxford UCI Corpus; PD = Parkinson's Disease; HC = Healthy Control; LOSO = Leave-One-Subject-Out; LOPO = Leave-One-Person-Out. Our results report mean &plusmn; standard deviation across 5 random seeds under strict 5-fold patient-wise GroupKFold.</div>
  </div>
</div>

<div class="full-width-float">
  <div class="ieee-fig">
    <img src="{fig_dir}/Fig2_MultiDataset_ROC_Curves.png" style="max-height: 3.35in;" alt="Fig. 2 Multi-Dataset ROC Curves and Calibration">
    <div class="ieee-fig-cap"><b>Fig. 2.</b> Receiver operating characteristic (ROC) curves and specificity calibration across all three cohorts. <b>(a)</b> Turkish UCI (TR, <i>N</i> = 756 recordings, <i>S</i> = 252 subjects). <b>(b)</b> Bengali BenSParX (BN, <i>N</i> = 900 recordings, <i>S</i> = 120 subjects). <b>(c)</b> English Oxford (OX, <i>N</i> = 195 recordings, <i>S</i> = 32 subjects). <b>(d)</b> Operating-point comparison showing how inner-fold Youden's <i>&tau;</i><sup>*</sup> threshold calibration increases Specificity (Spec) and Balanced Accuracy (BAcc) on 3:1 imbalanced cohorts (TR Spec rises from 51.5% to 71.5%; OX Spec rises from 39.6% to 91.7%), while Subject-Level Voting across replicate recordings achieves 86.3% (TR), 100.0% (BN), and 81.2% (OX) accuracy.</div>
  </div>
</div>

<div class="two-col">
  <h2 class="ieee-sec">IV. Experimental Results and Discussion</h2>
  <h3 class="ieee-subsec">A. Patient-Wise Performance and Literature Comparison</h3>
  <p class="no-indent">Table II reports the five-seed patient-wise evaluation across TR, BN, and OX, while Fig. 2 displays the corresponding ROC curves and threshold calibration trade-offs. On the high-dimensional Turkish corpus (<i>D</i> = 753), increasing the LightGBM learning rate from <i>&eta;</i> = 0.05 to <i>&eta;</i> = 0.10 improves five-seed recording-level accuracy from 84.18 &plusmn; 0.70% (ROC-AUC = 0.8713, PR-AUC = 0.9492) to 84.34 &plusmn; 0.27% (ROC-AUC = 0.8726, PR-AUC = 0.9484). Combining LightGBM (<i>&eta;</i> = 0.10) with KAN yields 85.00 &plusmn; 0.54% under nested inner-CV weight selection (ROC-AUC = 0.8740, PR-AUC = 0.9508) and 85.08 &plusmn; 0.49% at <i>&alpha;</i> = 0.60. When predictions are aggregated across the three vowel recordings per speaker, subject-level accuracy reaches 86.27 &plusmn; 0.74% (ROC-AUC = 0.8837). This bridges the apparent gap with prior leak-free Turkish studies: Sakar <i>et al.</i> [9] reported 86.0% under LOSO (which trains on 99.6% of subjects per fold rather than 80% in 5-fold CV), and G&uuml;nd&uuml;z [10] reported 86.9% using subject-level LOPO pooling.</p>

  <p>On the balanced Bengali conversational corpus (<i>N</i> = 900, <i>S</i> = 120), standalone LightGBM (<i>&eta;</i> = 0.10) achieves 92.84 &plusmn; 0.21% and KAN achieves 90.49 &plusmn; 0.41%. Their convex combination reaches 93.49 &plusmn; 0.36% across five seeds (and 93.56% on seed 42), matching the 93.78% single-split accuracy reported by Hossain <i>et al.</i> [11]. Moreover, because each Bengali participant contributes 7&ndash;8 conversational segments, subject-level probability voting classifies 100.00% of the 120 subjects correctly across all five seeds (ROC-AUC = 1.0000, Brier score 0.0167). On the smaller English Oxford cohort (<i>S</i> = 32), recording-level accuracy is 76.82 &plusmn; 3.33% for fusion (78.87 &plusmn; 5.12% for LightGBM <i>&eta;</i> = 0.05), rising to 81.25 &plusmn; 3.95% under subject-level voting.</p>

  <p><i>Addressing Low Specificity on Imbalanced Cohorts:</i> As shown in Table II and Fig. 2(d), evaluating TR at the unadjusted threshold <i>&tau;</i> = 0.50 yields 96.52% sensitivity but only 51.46% specificity (73.99% balanced accuracy, BAcc), flagging nearly half of healthy controls as PD due to the 3:1 class skew. Applying inner-fold Youden's <i>J</i> threshold calibration (<i>&tau;</i><sup>*</sup> &approx; 0.74) increases TR specificity by +20.00% to 71.46 &plusmn; 3.12% and improves balanced accuracy to 77.70 &plusmn; 1.15% while maintaining 83.94% sensitivity.</p>

  <h3 class="ieee-subsec">B. Comparison with Deep Tabular Baselines &amp; Ensemble Partners</h3>
  <p class="no-indent">Figure 3(a) compares our approach against ten tabular baselines on the 753-feature Turkish dataset. Deep attention and tree-mimetic architectures&mdash;including TabNet (71.96%), TabTransformer (75.91%), FT-Transformer (78.84%), GATE (80.02%), SAINT (80.41%), and NODE (81.61%)&mdash;underperform gradient-boosted trees when trained on <i>S</i> &approx; 200 subjects with <i>D</i> = 753 features, confirming that self-attention layers over-parameterize small-sample tabular clinical cohorts [25].</p>

  <p>A critical question is whether KAN contributes anything distinct compared to simpler non-linear ensemble partners such as an MLP, an RBF-kernel SVM, or logistic regression stacking. Figure 3(b) and Table II evaluate these exact ablations across all five seeds. On Bengali, <code>LightGBM + MLP</code> achieves the highest recording-level accuracy (94.16 &plusmn; 0.53% vs. 93.49 &plusmn; 0.36% for <code>LightGBM + KAN</code>), while on Turkish <code>LightGBM + SVM</code> (85.03%), <code>Stacking</code> (84.92%), <code>LightGBM + MLP</code> (84.89%), and <code>LightGBM + KAN</code> (85.08%) perform within 0.2% of one another in hard threshold accuracy. However, two differences favor the <code>LightGBM + KAN</code> pairing: 1) <i>Probability Ranking Stability:</i> On TR, pairing LightGBM with SVM degrades continuous probability ranking (ROC-AUC drops from 0.8726 to 0.8363) due to Platt-scaling variance on imbalanced folds, and Logistic Stacking degrades Oxford ROC-AUC to 0.4999, whereas <code>LightGBM + KAN</code> achieves the highest ROC-AUC (0.8740 on TR, 0.9784 on BN) and PR-AUC (0.9508 on TR, 0.9789 on BN); and 2) <i>Direct Univariate Interpretability:</i> Unlike black-box MLP hidden layers or RBF kernels, KAN's cubic B-spline edges directly expose smooth partial log-odds response curves &Delta;<i>&phi;<sub>i</sub></i>(<i>z</i>) for clinical auditing (Section IV-D).</p>
</div>

<div class="full-width-float">
  <div class="ieee-table-wrap">
    <div class="ieee-table-title"><b>Table II</b><br>Multi-Seed Patient-Wise Performance and Ensemble Partner Ablation Across Five Random Seeds (Mean &plusmn; SD)</div>
    <table class="ieee-table">
      <thead>
        <tr>
          <th class="l">Corpus</th>
          <th class="l">Model / Ensemble Configuration</th>
          <th>Acc (%)</th>
          <th>Sens (%)</th>
          <th>Spec (%)</th>
          <th>BAcc (%)</th>
          <th>F1 (%)</th>
          <th>MCC</th>
          <th>ROC-AUC</th>
          <th>PR-AUC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="l"><b>TR</b></td>
          <td class="l">Standalone LightGBM (<i>&eta;</i> = 0.05, 100 trees)</td>
          <td>84.18 &plusmn; 0.70</td>
          <td>95.50 &plusmn; 0.45</td>
          <td>50.94 &plusmn; 2.80</td>
          <td>73.22 &plusmn; 1.48</td>
          <td>90.01 &plusmn; 0.41</td>
          <td>0.547 &plusmn; 0.025</td>
          <td>0.8713 &plusmn; 0.0029</td>
          <td>0.9492 &plusmn; 0.0018</td>
        </tr>
        <tr>
          <td class="l"><b>TR</b></td>
          <td class="l">Standalone LightGBM (<i>&eta;</i> = 0.10, stronger baseline)</td>
          <td>84.34 &plusmn; 0.27</td>
          <td>95.39 &plusmn; 0.35</td>
          <td>51.88 &plusmn; 1.82</td>
          <td>73.63 &plusmn; 0.81</td>
          <td>90.09 &plusmn; 0.15</td>
          <td>0.553 &plusmn; 0.010</td>
          <td>0.8726 &plusmn; 0.0031</td>
          <td>0.9484 &plusmn; 0.0019</td>
        </tr>
        <tr>
          <td class="l"><b>TR</b></td>
          <td class="l">Standalone KAN (Cubic B-Spline [753, 8, 2])</td>
          <td>82.09 &plusmn; 0.11</td>
          <td>94.15 &plusmn; 0.24</td>
          <td>46.67 &plusmn; 0.78</td>
          <td>70.41 &plusmn; 0.34</td>
          <td>88.69 &plusmn; 0.07</td>
          <td>0.482 &plusmn; 0.005</td>
          <td>0.8334 &plusmn; 0.0154</td>
          <td>0.9299 &plusmn; 0.0136</td>
        </tr>
        <tr>
          <td class="l"><b>TR</b></td>
          <td class="l">Ablation: LightGBM (<i>&eta;</i> = 0.10) + MLP (2-Layer ReLU)</td>
          <td>84.89 &plusmn; 0.36</td>
          <td>95.50 &plusmn; 0.41</td>
          <td>53.75 &plusmn; 1.50</td>
          <td>74.62 &plusmn; 0.77</td>
          <td>90.41 &plusmn; 0.22</td>
          <td>0.571 &plusmn; 0.013</td>
          <td>0.8680 &plusmn; 0.0052</td>
          <td>0.9478 &plusmn; 0.0028</td>
        </tr>
        <tr>
          <td class="l"><b>TR</b></td>
          <td class="l">Ablation: LightGBM (<i>&eta;</i> = 0.10) + SVM (RBF Kernel)</td>
          <td>85.03 &plusmn; 0.28</td>
          <td>95.99 &plusmn; 0.32</td>
          <td>52.81 &plusmn; 1.40</td>
          <td>74.40 &plusmn; 0.68</td>
          <td>90.54 &plusmn; 0.17</td>
          <td>0.574 &plusmn; 0.010</td>
          <td>0.8363 &plusmn; 0.0061</td>
          <td>0.9241 &plusmn; 0.0054</td>
        </tr>
        <tr>
          <td class="l"><b>TR</b></td>
          <td class="l">Ablation: Logistic Regression Meta-Stacking</td>
          <td>84.92 &plusmn; 0.26</td>
          <td>95.78 &plusmn; 0.35</td>
          <td>53.02 &plusmn; 1.25</td>
          <td>74.40 &plusmn; 0.62</td>
          <td>90.46 &plusmn; 0.16</td>
          <td>0.571 &plusmn; 0.009</td>
          <td>0.8504 &plusmn; 0.0102</td>
          <td>0.9193 &plusmn; 0.0110</td>
        </tr>
        <tr>
          <td class="l"><b>TR</b></td>
          <td class="l">Proposed: LightGBM + KAN (Nested Inner-CV <i>&alpha;</i><sup>*</sup>)</td>
          <td>85.00 &plusmn; 0.54</td>
          <td>96.31 &plusmn; 0.48</td>
          <td>51.77 &plusmn; 1.95</td>
          <td>74.04 &plusmn; 1.02</td>
          <td>90.55 &plusmn; 0.33</td>
          <td>0.572 &plusmn; 0.018</td>
          <td><b>0.8740 &plusmn; 0.0029</b></td>
          <td><b>0.9508 &plusmn; 0.0020</b></td>
        </tr>
        <tr class="highlight-row">
          <td class="l"><b>TR</b></td>
          <td class="l">Proposed: LightGBM + KAN (<i>&alpha;</i> = 0.60, Recording-Level)</td>
          <td><b>85.08 &plusmn; 0.49</b></td>
          <td><b>96.52 &plusmn; 0.42</b></td>
          <td>51.46 &plusmn; 1.85</td>
          <td>73.99 &plusmn; 0.96</td>
          <td><b>90.61 &plusmn; 0.30</b></td>
          <td><b>0.575 &plusmn; 0.016</b></td>
          <td>0.8738 &plusmn; 0.0028</td>
          <td>0.9507 &plusmn; 0.0019</td>
        </tr>
        <tr>
          <td class="l"><b>TR</b></td>
          <td class="l">Proposed: LightGBM + KAN (In-Fold Youden's <i>&tau;</i><sup>*</sup> Calibrated)</td>
          <td>80.77 &plusmn; 0.61</td>
          <td>83.94 &plusmn; 1.12</td>
          <td><b>71.46 &plusmn; 3.12</b></td>
          <td><b>77.70 &plusmn; 1.15</b></td>
          <td>86.69 &plusmn; 0.48</td>
          <td>0.526 &plusmn; 0.015</td>
          <td>0.8738 &plusmn; 0.0028</td>
          <td>0.9507 &plusmn; 0.0019</td>
        </tr>
        <tr class="highlight-row">
          <td class="l"><b>TR</b></td>
          <td class="l">Proposed: LightGBM + KAN (Subject-Level Voting, <i>S</i> = 252)</td>
          <td><b>86.27 &plusmn; 0.74</b></td>
          <td><b>97.34 &plusmn; 0.47</b></td>
          <td>53.75 &plusmn; 2.86</td>
          <td>75.55 &plusmn; 1.48</td>
          <td><b>91.36 &plusmn; 0.44</b></td>
          <td><b>0.612 &plusmn; 0.025</b></td>
          <td><b>0.8837 &plusmn; 0.0039</b></td>
          <td><b>0.9539 &plusmn; 0.0027</b></td>
        </tr>
        <tr class="sec-divider">
          <td class="l"><b>BN</b></td>
          <td class="l">Standalone LightGBM (<i>&eta;</i> = 0.05 / <i>&eta;</i> = 0.10)</td>
          <td>92.11 / 92.84</td>
          <td>93.07 / 93.24</td>
          <td>91.16 / 92.44</td>
          <td>92.11 / 92.84</td>
          <td>92.19 / 92.87</td>
          <td>0.842 / 0.857</td>
          <td>0.9718 / 0.9772</td>
          <td>0.9703 / 0.9770</td>
        </tr>
        <tr>
          <td class="l"><b>BN</b></td>
          <td class="l">Standalone KAN (Cubic B-Spline [71, 8, 2])</td>
          <td>90.49 &plusmn; 0.41</td>
          <td>87.78 &plusmn; 0.67</td>
          <td>93.20 &plusmn; 0.91</td>
          <td>90.49 &plusmn; 0.41</td>
          <td>90.22 &plusmn; 0.41</td>
          <td>0.811 &plusmn; 0.008</td>
          <td>0.9621 &plusmn; 0.0044</td>
          <td>0.9658 &plusmn; 0.0069</td>
        </tr>
        <tr>
          <td class="l"><b>BN</b></td>
          <td class="l">Ablation: LightGBM + MLP / LightGBM + SVM / Stacking</td>
          <td><b>94.16</b> / 93.02 / 93.31</td>
          <td>92.31 / 92.71 / 93.02</td>
          <td><b>96.00</b> / 93.33 / 93.60</td>
          <td><b>94.16</b> / 93.02 / 93.31</td>
          <td><b>94.04</b> / 93.00 / 93.29</td>
          <td><b>0.884</b> / 0.860 / 0.866</td>
          <td><b>0.9858</b> / 0.9761 / 0.9727</td>
          <td><b>0.9857</b> / 0.9775 / 0.9709</td>
        </tr>
        <tr class="highlight-row">
          <td class="l"><b>BN</b></td>
          <td class="l">Proposed: LightGBM + KAN (Recording-Level, <i>N</i> = 900)</td>
          <td>93.49 &plusmn; 0.36</td>
          <td>92.53 &plusmn; 0.41</td>
          <td>94.44 &plusmn; 0.51</td>
          <td>93.49 &plusmn; 0.36</td>
          <td>93.43 &plusmn; 0.36</td>
          <td>0.870 &plusmn; 0.007</td>
          <td>0.9784 &plusmn; 0.0016</td>
          <td>0.9789 &plusmn; 0.0020</td>
        </tr>
        <tr class="highlight-row">
          <td class="l"><b>BN</b></td>
          <td class="l">Proposed: LightGBM + KAN (Subject-Level Voting, <i>S</i> = 120)</td>
          <td><b>100.00 &plusmn; 0.00</b></td>
          <td><b>100.00 &plusmn; 0.00</b></td>
          <td><b>100.00 &plusmn; 0.00</b></td>
          <td><b>100.00 &plusmn; 0.00</b></td>
          <td><b>100.00 &plusmn; 0.00</b></td>
          <td><b>1.000 &plusmn; 0.000</b></td>
          <td><b>1.0000 &plusmn; 0.0000</b></td>
          <td><b>1.0000 &plusmn; 0.0000</b></td>
        </tr>
        <tr class="sec-divider">
          <td class="l"><b>OX</b></td>
          <td class="l">Standalone LightGBM (<i>&eta;</i> = 0.05) / Standalone KAN</td>
          <td>78.87 / 75.08</td>
          <td>93.06 / 88.30</td>
          <td>35.42 / 34.58</td>
          <td>64.24 / 61.44</td>
          <td>86.91 / 84.25</td>
          <td>0.343 / 0.249</td>
          <td>0.7168 / <b>0.7201</b></td>
          <td>0.8592 / <b>0.8949</b></td>
        </tr>
        <tr class="highlight-row">
          <td class="l"><b>OX</b></td>
          <td class="l">Proposed: LightGBM + KAN (Recording <i>N</i> = 195 / Subject <i>S</i> = 32)</td>
          <td>76.82 / <b>81.25</b></td>
          <td>90.88 / <b>98.33</b></td>
          <td>33.75 / 30.00</td>
          <td>62.32 / <b>64.17</b></td>
          <td>85.54 / <b>88.74</b></td>
          <td>0.286 / <b>0.404</b></td>
          <td>0.6819 / 0.6750</td>
          <td>0.8466 / 0.8526</td>
        </tr>
      </tbody>
    </table>
    <div class="table-note">Abbreviations: TR = Turkish UCI (<i>N</i> = 756, <i>S</i> = 252); BN = Bengali BenSParX (<i>N</i> = 900, <i>S</i> = 120); OX = English Oxford (<i>N</i> = 195, <i>S</i> = 32); Acc = Accuracy; Sens = Sensitivity (Recall); Spec = Specificity; BAcc = Balanced Accuracy; F1 = F1-Score; MCC = Matthews Correlation Coefficient; ROC-AUC = Area Under ROC Curve; PR-AUC = Precision-Recall AUC.</div>
  </div>
</div>

<div class="full-width-float">
  <div class="ieee-fig">
    <img src="{fig_dir}/Fig3_Architecture_and_Ensemble_Ablation.png" style="max-height: 2.15in;" alt="Fig. 3 Architecture and Ensemble Partner Ablation">
    <div class="ieee-fig-cap"><b>Fig. 3.</b> Architectural and ensemble partner ablations under strict patient-wise five-fold GroupKFold. <b>(a)</b> Benchmark of 13 tabular architectures on the high-dimensional Turkish corpus (TR, <i>D</i> = 753, 5-seed mean &plusmn; SD), showing that gradient-boosted trees and tree-spline fusion outperform deep attention models (TabNet, TabTransformer, FT-Transformer, GATE, SAINT, NODE). <b>(b)</b> Five-seed ensemble partner ablation across TR, Bengali (BN), and Oxford (OX) comparing standalone models against LightGBM+SVM, LightGBM+MLP, Logistic Regression Stacking, recording-level LightGBM+KAN, and subject-level voting.</div>
  </div>
</div>

<div class="full-width-float">
  <div class="ieee-fig">
    <img src="{fig_dir}/Fig4_Leakage_and_Feature_Sensitivity.png" style="max-height: 1.85in;" alt="Fig. 4 Leakage Inflation and Sensitivity">
    <div class="ieee-fig-cap"><b>Fig. 4.</b> Methodological sensitivity studies across five random seeds. <b>(a)</b> Quantification of speaker identity leakage comparing five-fold record-wise stratified splitting against patient-wise GroupKFold: record-wise splitting inflates accuracy by +6.56% on Turkish (TR, 90.90% vs. 84.34%), +0.27% on Bengali (BN, 93.11% vs. 92.84%), and +16.82% on English Oxford (OX, 93.64% vs. 76.82%). <b>(b)</b> Accuracy trajectory on TR across ANOVA feature subset sizes <i>K</i> &isin; {50, 100, 200, 500, 753}. <b>(c)</b> Out-of-fold accuracy across convex mixing weights <i>&alpha;</i> &isin; [0, 1], peaking at <i>&alpha;</i> = 0.60.</div>
  </div>
</div>

<div class="two-col">
  <h3 class="ieee-subsec">C. Statistical Significance and Speaker Identity Leakage</h3>
  <p class="no-indent">Figure 4(a) quantifies the exact accuracy inflation caused by record-wise splitting across five seeds using the same LightGBM (<i>&eta;</i> = 0.10) classifier. When replicate recordings from the same patient are split across training and test sets, accuracy is artificially inflated by +6.56% on Turkish (90.90 &plusmn; 0.55% record-wise vs. 84.34 &plusmn; 0.27% patient-wise; +4.63% to +4.89% for <i>&eta;</i> = 0.05 on single seeds) and by +16.82% on Oxford (93.64 &plusmn; 0.25% vs. 76.82 &plusmn; 3.39%). Figure 4(b) further shows that pruning the 753 Turkish features down to <i>K</i> = 50 drops accuracy to 80.40%, demonstrating that high-frequency TQWT sub-bands contribute useful complementary signal.</p>

  <p>Table III addresses the non-independence of repeated recordings in statistical testing. In a single five-fold cross-validation experiment (<i>N</i><sub>folds</sub> = 5), a two-sided Wilcoxon signed-rank test has a mathematical minimum achievable <i>p</i>-value of 2 &times; (0.5)<sup>5</sup> = 0.0625 even when one model wins all five folds, which explains why five-fold Wilcoxon tests cannot reach <i>p</i> &lt; 0.05. Evaluating all 25 paired folds across our five random seeds overcomes this sample-size limit: <code>LightGBM + KAN</code> outperforms the stronger <code>LightGBM (&eta;=0.10)</code> baseline with <i>p</i> = 0.0453 on TR (and <i>p</i> = 0.0129 against <i>&eta;</i> = 0.05) and outperforms standalone KAN with <i>p</i> = 3.5 &times; 10<sup>&minus;5</sup> on TR and <i>p</i> = 4.8 &times; 10<sup>&minus;5</sup> on BN. Under Obuchowski's patient-clustered test (Eq. 4), the gain over standalone KAN remains statistically significant after accounting for intra-subject correlation (<i>&chi;</i><sup>2</sup><sub>clus</sub> = 9.00, <i>p</i> = 0.0027 on TR; <i>&chi;</i><sup>2</sup><sub>clus</sub> = 9.52, <i>p</i> = 0.0020 on BN), whereas recording-level differences among the top tree ensembles (<code>LGBM+KAN</code> vs. <code>LGBM+MLP</code> or <code>LGBM (&eta;=0.10)</code>) are not statistically significant at the subject level (<i>p</i> &gt; 0.40).</p>

  <h3 class="ieee-subsec">D. Explainability and Cross-Lingual Acoustic Polarity Flip</h3>
  <p class="no-indent">Figures 5(a)&ndash;(b) display the top predictors ranked by TreeSHAP on TR and Welch's |<i>t</i>|-statistic on BN. In both cohorts, dynamic log-energy acceleration (<code>std_delta_delta_log_energy</code>, capturing glottal micro-tremor) and second-order cepstral derivatives rank at the top. However, a feature ranking highly in two languages does not mean its directional effect is invariant across speech tasks. Figure 5(c) plots the standardized effect size (Cohen's <i>d</i> = (<i>&mu;</i><sub>PD</sub> &minus; <i>&mu;</i><sub>HC</sub>) / <i>&sigma;</i><sub>pooled</sub>) across TR sustained vowels and BN spontaneous conversation:</p>
  <p>&bull; <i>Directionally Consistent Biomarkers:</i> Static second cepstral coefficient <code>MFCC_2</code> is elevated in PD in both cohorts (TR <code>mean_MFCC_2nd_coef</code>: PD 1.65 &plusmn; 1.44 vs. HC 0.13 &plusmn; 1.77, <i>d</i> = +0.95; BN <code>MFCC_2</code>: PD 0.215 &plusmn; 0.185 vs. HC 0.019 &plusmn; 0.223, <i>d</i> = +0.96). Likewise, <code>std_delta_delta_log_energy</code> (<i>d</i> = +0.87 TR, +0.90 BN), <code>locPctJitter</code> (<i>d</i> = +0.55 TR, +0.61 BN), and <code>locShimmer</code> (<i>d</i> = +0.38 TR, +0.43 BN) consistently increase in PD, while first formant <code>F1</code> (<i>d</i> = &minus;0.73 TR, &minus;0.67 BN) and <code>HNR</code> (<i>d</i> = &minus;0.50 TR, &minus;0.02 BN) decrease.</p>
  <p>&bull; <i>Task-Induced Polarity Inversion (<code>Delta_MFCC_2</code> &amp; <code>Delta_MFCC_6</code>):</i> First-order cepstral velocity <code>Delta_MFCC_2</code> flips sign between the two corpora: in Turkish sustained vowels (<code>/a/</code>), <code>mean_2nd_delta</code> is slightly <i>lower</i> in PD than HC (<i>d</i> = &minus;0.24, <i>p</i> = 0.013), whereas in Bengali conversation, <code>Delta_MFCC_2</code> is strongly <i>elevated</i> in PD (1.79 &plusmn; 1.54 vs. 0.16 &plusmn; 1.86, <i>d</i> = +0.96, <i>p</i> &lt; 10<sup>&minus;36</sup>). Conversely, <code>Delta_MFCC_6</code> shifts from <i>d</i> = +0.05 in TR to <i>d</i> = &minus;0.19 in BN. Physically, sustaining a stationary vowel requires holding the vocal tract still, where PD rigidity reduces slow spectral drift; conversational speech requires rapid consonant-to-vowel transitions, where PD articulatory undershoot and tremor introduce irregular frame-to-frame spectral jumps.</p>
</div>

<div class="full-width-float">
  <div class="ieee-table-wrap">
    <div class="ieee-table-title"><b>Table III</b><br>Recording-Level vs. Patient-Clustered Statistical Hypothesis Testing (Accounting for Repeated Recordings per Subject)</div>
    <table class="ieee-table">
      <thead>
        <tr>
          <th class="l">Corpus (<i>N</i> rec, <i>S</i> subj)</th>
          <th class="l">Paired Model Comparison (Proposed LightGBM+KAN vs. Baseline)</th>
          <th>Recording Discordant (<i>b</i> : <i>c</i>)</th>
          <th>Unadjusted Rec. McNemar <i>p</i></th>
          <th>Obuchowski Clustered <i>&chi;</i><sup>2</sup><sub>clus</sub> (<i>p</i>-val)</th>
          <th>Subject-Level Discordant (<i>b<sub>S</sub></i> : <i>c<sub>S</sub></i>)</th>
          <th>Subject McNemar <i>p</i></th>
          <th>25-Fold (5-Seed) Wilcoxon <i>p</i></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="l"><b>TR</b> (<i>N</i> = 756, <i>S</i> = 252)</td>
          <td class="l">Proposed Fusion vs. Stronger LightGBM (<i>&eta;</i> = 0.10)</td>
          <td>16 : 13</td>
          <td>0.7111</td>
          <td>0.23 (0.6315)</td>
          <td>5 : 3</td>
          <td>0.7266</td>
          <td><b>0.0453*</b></td>
        </tr>
        <tr>
          <td class="l"><b>TR</b> (<i>N</i> = 756, <i>S</i> = 252)</td>
          <td class="l">Proposed Fusion vs. Baseline LightGBM (<i>&eta;</i> = 0.05)</td>
          <td>16 : 18</td>
          <td>0.8642</td>
          <td>0.11 (0.7393)</td>
          <td>5 : 1</td>
          <td>0.2188</td>
          <td><b>0.0129*</b></td>
        </tr>
        <tr>
          <td class="l"><b>TR</b> (<i>N</i> = 756, <i>S</i> = 252)</td>
          <td class="l">Proposed Fusion vs. Standalone KAN (Spline)</td>
          <td>33 : 9</td>
          <td><b>0.0003*</b></td>
          <td><b>9.00 (0.0027*)</b></td>
          <td>7 : 3</td>
          <td>0.3438</td>
          <td><b>0.00004*</b></td>
        </tr>
        <tr>
          <td class="l"><b>TR</b> (<i>N</i> = 756, <i>S</i> = 252)</td>
          <td class="l">Proposed Fusion vs. LightGBM + MLP / LightGBM + SVM</td>
          <td>7:6 / 13:14</td>
          <td>1.000 / 1.000</td>
          <td>0.07 (0.80) / 0.03 (0.87)</td>
          <td>3:1 / 3:2</td>
          <td>0.625 / 1.000</td>
          <td>0.4105 / 0.9197</td>
        </tr>
        <tr class="sec-divider">
          <td class="l"><b>BN</b> (<i>N</i> = 900, <i>S</i> = 120)</td>
          <td class="l">Proposed Fusion vs. Stronger LightGBM (<i>&eta;</i> = 0.10) / (<i>&eta;</i> = 0.05)</td>
          <td>21:18 / 26:16</td>
          <td>0.749 / 0.164</td>
          <td>0.22 (0.64) / 1.92 (0.17)</td>
          <td>0:0 / 0:0 (100% Subj)</td>
          <td>1.000</td>
          <td>0.0784 / <b>0.0058*</b></td>
        </tr>
        <tr>
          <td class="l"><b>BN</b> (<i>N</i> = 900, <i>S</i> = 120)</td>
          <td class="l">Proposed Fusion vs. Standalone KAN (Spline)</td>
          <td>31 : 11</td>
          <td><b>0.0029*</b></td>
          <td><b>9.52 (0.0020*)</b></td>
          <td>0 : 0 (100% Subj)</td>
          <td>1.000</td>
          <td><b>0.00005*</b></td>
        </tr>
        <tr class="sec-divider">
          <td class="l"><b>OX</b> (<i>N</i> = 195, <i>S</i> = 32)</td>
          <td class="l">Proposed Fusion vs. Standalone KAN / LightGBM (<i>&eta;</i> = 0.05)</td>
          <td>5:1 / 2:6</td>
          <td>0.219 / 0.289</td>
          <td>2.00 (0.16) / 2.00 (0.16)</td>
          <td>1:0 / 0:0</td>
          <td>1.000</td>
          <td><b>0.0154*</b> / 0.0560</td>
        </tr>
      </tbody>
    </table>
    <div class="table-note">Note: <i>b</i> and <i>c</i> denote discordant recording counts where Proposed Fusion is correct and Baseline is incorrect (<i>b</i>) or vice versa (<i>c</i>); <i>b<sub>S</sub></i> : <i>c<sub>S</sub></i> denotes discordant independent subjects after subject-level voting. Note that a single 5-fold Wilcoxon test has a mathematical lower bound of <i>p</i> &ge; 2 &times; (0.5)<sup>5</sup> = 0.0625; evaluating all 25 folds across 5 seeds provides sufficient statistical power (*<i>p</i> &lt; 0.05).</div>
  </div>
</div>

<div class="full-width-float">
  <div class="ieee-fig">
    <img src="{fig_dir}/Fig5_SHAP_and_Polarity_Analysis.png" style="max-height: 1.92in;" alt="Fig. 5 TreeSHAP and Polarity Analysis">
    <div class="ieee-fig-cap"><b>Fig. 5.</b> Cross-lingual feature attribution and directional polarity analysis. <b>(a)</b> Top 10 Turkish (TR, sustained vowel /a/) features ranked by mean absolute TreeSHAP log-odds impact. <b>(b)</b> Top 10 Bengali (BN, spontaneous conversation) features ranked by absolute Welch |<i>t</i>|-statistic (<i>p</i> &lt; 10<sup>&minus;12</sup>). <b>(c)</b> Standardized effect size Cohen's <i>d</i> = (<i>&mu;</i><sub>PD</sub> &minus; <i>&mu;</i><sub>HC</sub>) / <i>&sigma;</i><sub>pooled</sub> across TR and BN. While static <code>MFCC_2</code>, <code>std_delta_delta_log_energy</code>, <code>locPctJitter</code>, <code>locShimmer</code>, and <code>F1</code> maintain consistent sign across languages, dynamic cepstral velocity features (<code>Delta_MFCC_2</code> and <code>Delta_MFCC_6</code>) flip sign between stationary vowel phonation and connected conversational speech.</div>
  </div>
</div>

<div class="full-width-float">
  <div class="ieee-fig">
    <img src="{fig_dir}/Fig6_KAN_Spline_Activations.png" style="max-height: 2.65in;" alt="Fig. 6 KAN Cubic B-Spline Response Curves">
    <div class="ieee-fig-cap"><b>Fig. 6.</b> Centered univariate cubic B-spline partial log-odds response curves &Delta;<i>&phi;<sub>i</sub></i>(<i>z</i>) = <i>&phi;<sub>i</sub></i>(<i>z</i>) &minus; <i>&phi;<sub>i</sub></i>(0) learned by the KAN branch across six core acoustic biomarkers on the Turkish (TR) cohort. Solid red lines and shaded bands indicate the mean and &plusmn;1 SD across five random seeds; bottom rug ticks show the empirical <i>z</i>-score distributions for Healthy Controls (HC, blue) and Parkinson's Disease (PD, red) subjects.</div>
  </div>
</div>

<div class="full-width-float">
  <div class="ieee-fig">
    <img src="{fig_dir}/Fig7_CrossLingual_Learning_Curves.png" style="max-height: 1.88in;" alt="Fig. 7 Few-Shot Cross-Lingual Learning Curves">
    <div class="ieee-fig-cap"><b>Fig. 7.</b> Few-shot cross-lingual learning curves across labeled target patient budgets (0%, 5%, 10%, 15%, 20%, 30%, and 50% of target subjects; 5-seed mean &plusmn; SD) comparing Target-Only training from scratch, Source+Target Transfer without CORAL, and CORAL + Source+Target Transfer against the native full-feature upper bound (green dotted line): <b>(a)</b> Turkish (TR) &rarr; Bengali (BN, 27 shared features), <b>(b)</b> BN &rarr; TR (27 shared features), and <b>(c)</b> TR &rarr; English Oxford (OX, 13 shared features).</div>
  </div>
</div>

<div class="two-col">
  <p class="no-indent">Figure 6 illustrates the centered univariate B-spline log-odds trajectories &Delta;<i>&phi;<sub>i</sub></i>(<i>z</i>) learned by KAN across five random seeds. Rather than imposing a linear log-odds slope, the splines capture clinically meaningful non-linear regimes: <code>mean_MFCC_2nd_coef</code> and <code>std_8th_delta_delta</code> remain near zero for healthy negative <i>z</i>-scores before rising steeply for <i>z</i> &gt; 0.3, whereas <code>tqwt_entropy_log_dec_12</code> exhibits a monotonic negative slope above <i>z</i> = 0 reflecting loss of harmonic periodicity.</p>

  <h3 class="ieee-subsec">E. Practical Scope and Limits of Cross-Lingual Transfer</h3>
  <p class="no-indent">Because 27 shared acoustic features represent only a subset of the 71 Bengali and 753 Turkish features&mdash;and because cepstral velocities flip sign between sustained vowels and conversation&mdash;it is important to evaluate cross-lingual transfer realistically. Figure 7 plots five-seed target patient learning curves from 0% (zero-shot) to 50% labeled target patients:</p>
  <p><i>1) When Target Data Are Abundant (&ge;20% Patients), Native Training Dominates:</i> Training a native classifier on all 71 Bengali features achieves 93.49%, whereas transferring a 27-feature model from Turkish with 20% labeled Bengali patients (<i>S</i><sub>cal</sub> = 24) achieves 84.43 &plusmn; 1.19% without CORAL and 83.79 &plusmn; 1.59% with CORAL (and 89.55% at 50% target budget). At 20%+ target budgets, labeled target samples already anchor the decision boundary, so unsupervised CORAL adds at most +0.45% to +1.17% (e.g., 88.22% vs. 87.05% at 30% BN budget) and can slightly over-rotate small target subsets.</p>
  <p><i>2) Cold-Start Utility at 0%&ndash;10% Target Budgets:</i> Cross-lingual transfer is practically useful only when commissioning a screening tool in a new clinic with very few labeled local patients. At a 5% target budget (<i>S</i><sub>cal</sub> = 6 Bengali patients, 12 Turkish patients, or 2 Oxford patients), training from scratch on target data alone yields 68.27 &plusmn; 1.80% on BN, 73.16 &plusmn; 4.30% on TR, and 55.63 &plusmn; 14.94% on OX. Initializing with the source-language ensemble boosts 5%-budget accuracy by +10.31% on BN (78.58%), +5.49% on TR (78.65%), and +13.00% on OX (68.63% unaligned; 72.57% with CORAL). Similarly, in the purely zero-shot regime (0% labeled target data), unsupervised CORAL alignment improves <code>BN &rarr; TR</code> accuracy from 82.01% to 85.19% (ROC-AUC = 0.9425) and <code>TR &rarr; OX</code> from 70.26% to 71.28%.</p>

  <h3 class="ieee-subsec">F. Limitations</h3>
  <p class="no-indent">Several limitations warrant consideration. First, while five-fold patient-grouped validation prevents speaker leakage, the English Oxford cohort contains only 32 subjects (8 HC), leading to higher fold-to-fold variance (&plusmn;3.33%). Second, the Turkish and Oxford corpora rely on sustained <code>/a/</code> phonations recorded in quiet settings, whereas BenSParX uses conversational speech; standardized multi-center corpora recording identical protocols across languages are needed. Third, no UPDRS severity scores or longitudinal medication states (<code>ON</code>/<code>OFF</code> levodopa) were available across all three datasets to stratify early-stage prodromal cases.</p>

  <h2 class="ieee-sec">V. Conclusion and Code Availability</h2>
  <p class="no-indent">We presented a multi-seed, patient-grouped evaluation of gradient-boosted decision trees and Kolmogorov&ndash;Arnold spline networks across Turkish, Bengali, and English Parkinson's speech cohorts. Enforcing strict speaker disjointness revealed that record-wise splitting inflates accuracy by 6.6% to 16.8%. Combining LightGBM and KAN probabilities with inner-fold Youden threshold calibration improved specificity on imbalanced cohorts from ~51% to over 71%, while subject-level voting across replicate recordings achieved 86.27% accuracy on Turkish, 100.00% on Bengali, and 81.25% on Oxford. Finally, cross-lingual analysis showed that while task-induced polarity flips in dynamic MFCC derivatives make native target-language training preferable whenever sufficient local data exist, source-model transfer provides a 5.5% to 13.0% cold-start advantage when fewer than 10% of target patients are labeled.</p>
  <p><b>Code and Data Availability:</b> All training scripts, five-seed cross-validation splits, homologous feature mappings, and figure generation pipelines are publicly available at <u>https://github.com/Arunavo-build/Perkinson-Disease</u>. Extended mathematical derivations, baseline hyperparameter grids, the 27-feature mapping table, confusion matrices (Fig. S1), and training curves (Fig. S2) are provided in the accompanying Supplementary Material.</p>

  <h2 class="ieee-sec">References</h2>
  <div class="ref-list">
    __REFS_HTML__
  </div>
</div>

</body>
</html>
"""
    html = html.replace("{fig_dir}", fig_dir).replace("__REFS_HTML__", refs_html)
    html = convert_inline_code_and_math(html)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved main HTML to {HTML_PATH}")


def generate_supplementary_html(fig_dir, refs):
    df_map = pd.read_csv(os.path.join(WORKSPACE_DIR, "results", "homologous_feature_mapping.csv"))
    family_notes = {
        "Glottal Pulse Periodicity": "Stationary /a/ glottal cycles (75-500 Hz pitch floor/ceiling) vs. voiced-segment pulse gating across conversational prosody",
        "Fundamental Frequency Perturbation (Jitter)": "Cycle-to-cycle pitch perturbation on stationary vowel vs. voiced-frame autocorrelation peak picking in running speech",
        "Amplitude Perturbation (Shimmer)": "Peak-to-peak glottal amplitude stability (clinical condenser mic) vs. conversational syllable envelope (smartphone mic)",
        "Glottal Harmonic-to-Noise Ratio": "Harmonic periodicity over turbulent aspiration noise in sustained /a/ vs. voiced conversational frames",
        "Vocal Pitch & Subglottal Intensity": "Uncalibrated SPL dB range during steady phonation vs. utterance-level conversational intensity contour",
        "Vocal Tract Formant Frequencies & Bandwidths": "Burg LPC formant frequencies (F1-F4) on static open-mid vowel /a/ vs. coarticulated vowel transitions in Bengali",
        "Dynamic Log-Energy Kinematics": "Frame-level log-energy and 1st/2nd temporal derivatives (25 ms Hamming window, 10 ms hop; within-fold z-score normalized)"
    }
    map_rows = []
    for row in df_map.itertuples():
        spec_note = family_notes.get(row.Feature_Category, row.Extraction_Difference)
        map_rows.append(
            f"<tr><td>{row.Index + 1}</td><td class='l'>{row.Feature_Category}</td>"
            f"<td class='l'><code>{row.Turkish_UCI_Column}</code></td>"
            f"<td class='l'><code>{row.Bengali_BenSParX_Column}</code></td>"
            f"<td class='l'>{spec_note}</td></tr>"
        )
    map_table_html = "\n".join(map_rows)

    supp_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Supplementary Material: Patient-Grouped Parkinson's Disease Screening Across Languages</title>
<style>
  @page {
    size: letter portrait;
    margin: 0.48in 0.48in 0.48in 0.48in;
  }
  body {
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 9.0pt;
    line-height: 1.26;
    color: #000;
    margin: 0;
  }
  code {
    font-family: "Courier New", Courier, monospace;
    font-size: 8.1pt;
  }
  .header {
    font-size: 7.4pt;
    text-transform: uppercase;
    border-bottom: 0.8pt solid #000;
    padding-bottom: 3px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
  }
  h1 {
    font-size: 14.2pt;
    text-align: center;
    margin: 0 0 5px 0;
  }
  .sub {
    text-align: center;
    font-size: 9.2pt;
    margin-bottom: 12px;
    color: #222;
  }
  h2 {
    font-size: 10.2pt;
    font-variant: small-caps;
    border-bottom: 0.4pt solid #666;
    padding-bottom: 2px;
    margin: 10px 0 5px 0;
  }
  p {
    margin: 0 0 5px 0;
    text-align: justify;
  }
  .eq {
    display: flex;
    justify-content: space-between;
    margin: 4px 12px;
    font-size: 8.8pt;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 7.5pt;
    margin: 6px 0 10px 0;
    border-top: 1.1pt solid #000;
    border-bottom: 1.1pt solid #000;
  }
  th {
    border-bottom: 0.6pt solid #000;
    padding: 2.5px 4px;
    text-align: center;
    font-weight: bold;
  }
  td {
    padding: 2px 4px;
    text-align: center;
    border-bottom: 0.25pt solid #ddd;
  }
  td.l, th.l {
    text-align: left;
  }
  .fig {
    text-align: center;
    margin: 8px 0;
    break-inside: avoid;
  }
  .fig img {
    max-width: 95%;
    height: auto;
  }
  .cap {
    font-size: 8.0pt;
    text-align: justify;
    margin-top: 3px;
  }
</style>
</head>
<body>

<div class="header">
  <span>SUPPLEMENTARY MATERIAL &mdash; IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS</span>
  <span>ARUNAVO: PATIENT-GROUPED PD SPEECH SCREENING</span>
</div>

<h1>Supplementary Material: Patient-Grouped Parkinson's Disease Screening Across Languages via LightGBM and Kolmogorov&ndash;Arnold Spline Ensembles</h1>
<div class="sub">Arunavo &bull; Companion Document to Main Manuscript &bull; Repository: https://github.com/Arunavo-build/Perkinson-Disease</div>

<h2>Section S1. Standard Acoustic Signal Processing and B-Spline Equations</h2>
<p>This section provides the standard mathematical definitions for the acoustic feature families and B-spline basis functions referenced in Section III of the main text.</p>
<p><b>1) Fundamental Frequency Perturbation (Local and RAP Jitter):</b> Given <i>K</i> consecutive glottal pitch periods <i>T</i><sub>1</sub>, <i>T</i><sub>2</sub>, &hellip;, <i>T<sub>K</sub></i> extracted via autocorrelation peak picking, cycle-to-cycle local percentage jitter and three-point relative average perturbation (RAP) are defined as:</p>
<div class="eq">
  <span>Jitter<sub>loc(%)</sub> = [ (1 / (<i>K</i> &minus; 1)) &Sigma;<sub><i>i</i>=1</sub><sup><i>K</i>&minus;1</sup> |<i>T<sub>i</sub></i> &minus; <i>T</i><sub><i>i</i>+1</sub>| ] / [ (1 / <i>K</i>) &Sigma;<sub><i>i</i>=1</sub><sup><i>K</i></sup> <i>T<sub>i</sub></i> ] &times; 100, &emsp; RAP = [ (1 / (<i>K</i> &minus; 2)) &Sigma;<sub><i>i</i>=2</sub><sup><i>K</i>&minus;1</sup> |<i>T<sub>i</sub></i> &minus; (<i>T</i><sub><i>i</i>&minus;1</sub> + <i>T<sub>i</sub></i> + <i>T</i><sub><i>i</i>+1</sub>)/3| ] / [ (1 / <i>K</i>) &Sigma;<sub><i>i</i>=1</sub><sup><i>K</i></sup> <i>T<sub>i</sub></i> ].</span>
  <span>(S1)</span>
</div>

<p><b>2) Amplitude Perturbation (Local and APQ Shimmer):</b> Given peak-to-peak glottal cycle amplitudes <i>A</i><sub>1</sub>, <i>A</i><sub>2</sub>, &hellip;, <i>A<sub>K</sub></i>, local shimmer in decibels (dB) and the <i>q</i>-point amplitude perturbation quotient (APQ<sub><i>q</i></sub>, <i>q</i> &isin; {3, 5, 11}) are:</p>
<div class="eq">
  <span>Shimmer<sub>dB</sub> = (1 / (<i>K</i> &minus; 1)) &Sigma;<sub><i>i</i>=1</sub><sup><i>K</i>&minus;1</sup> |20 log<sub>10</sub>(<i>A</i><sub><i>i</i>+1</sub> / <i>A<sub>i</sub></i>)|, &emsp; APQ<sub><i>q</i></sub> = [ (1 / (<i>K</i> &minus; <i>q</i> + 1)) &Sigma;<sub><i>i</i>=1</sub><sup><i>K</i>&minus;<i>q</i>+1</sup> |<i>A<sub>i</sub></i> &minus; (1/<i>q</i>)&Sigma;<sub><i>m</i>=0</sub><sup><i>q</i>&minus;1</sup> <i>A</i><sub><i>i</i>+<i>m</i></sub>| ] / [ (1 / <i>K</i>) &Sigma;<sub><i>i</i>=1</sub><sup><i>K</i></sup> <i>A<sub>i</sub></i> ].</span>
  <span>(S2)</span>
</div>

<p><b>3) Harmonics-to-Noise Ratio (HNR) and Pitch Period Entropy (PPE):</b> For an acoustic frame <i>s</i>(<i>t</i>) with fundamental period <i>T</i><sub>0</sub>, HNR quantifies harmonic periodicity relative to turbulent glottal noise, while PPE measures the Shannon entropy of the pitch-period residual distribution <i>p<sub>r</sub></i> on a semitone scale:</p>
<div class="eq">
  <span>HNR (dB) = 10 log<sub>10</sub> [ <i>R</i>(<i>T</i><sub>0</sub>) / (<i>R</i>(0) &minus; <i>R</i>(<i>T</i><sub>0</sub>)) ], &emsp; PPE = &minus;&Sigma;<sub><i>r</i></sub> <i>p<sub>r</sub></i> log<sub>2</sub>(<i>p<sub>r</sub></i>).</span>
  <span>(S3)</span>
</div>

<p><b>4) Mel-Frequency Cepstral Coefficients (MFCCs) and Dynamic Velocity/Acceleration:</b> Using a <i>B</i>-channel triangular Mel filterbank <i>H<sub>b</sub></i>(<i>k</i>) applied to the short-time Fourier power spectrum |<i>S</i>(<i>k</i>)|<sup>2</sup>, static cepstral coefficients <i>c<sub>n</sub></i> and their first-order (&Delta;<i>c<sub>n</sub></i>, velocity) and second-order (&Delta;<sup>2</sup><i>c<sub>n</sub></i>, acceleration) temporal derivatives over window <i>W</i> = 2 are computed as:</p>
<div class="eq">
  <span><i>c<sub>n</sub></i> = &Sigma;<sub><i>b</i>=1</sub><sup><i>B</i></sup> log(&Sigma;<sub><i>k</i></sub> |<i>S</i>(<i>k</i>)|<sup>2</sup> <i>H<sub>b</sub></i>(<i>k</i>)) cos[&pi;<i>n</i>(<i>b</i> &minus; 0.5)/<i>B</i>], &emsp; &Delta;<i>c<sub>n</sub></i><sup>(<i>t</i>)</sup> = [ &Sigma;<sub><i>&tau;</i>=1</sub><sup><i>W</i></sup> <i>&tau;</i>(<i>c<sub>n</sub></i><sup>(<i>t</i>+<i>&tau;</i>)</sup> &minus; <i>c<sub>n</sub></i><sup>(<i>t</i>&minus;<i>&tau;</i>)</sup>) ] / [ 2 &Sigma;<sub><i>&tau;</i>=1</sub><sup><i>W</i></sup> <i>&tau;</i><sup>2</sup> ].</span>
  <span>(S4)</span>
</div>

<p><b>5) Tunable Q-Factor Wavelet Transform (TQWT):</b> TQWT decomposes the voice signal into <i>J</i> = 36 sub-bands parameterized by quality factor <i>Q</i>, redundancy <i>r</i>, and low/high-pass scaling factors <i>&alpha;</i> = 1 &minus; 2/[(<i>Q</i> + 1)<i>r</i>], <i>&beta;</i> = 2/(<i>Q</i> + 1). The sub-band log-energy entropy of level <i>j</i> with wavelet coefficients <i>w<sub>j,n</sub></i> is:</p>
<div class="eq">
  <span><i>E</i><sub>log, <i>j</i></sub> = &Sigma;<sub><i>n</i>=1</sub><sup><i>N<sub>j</sub></i></sup> log(<i>w<sub>j,n</sub></i><sup>2</sup> + <i>&epsilon;</i>), &emsp; TKEO(<i>w<sub>j,n</sub></i>) = <i>w<sub>j,n</sub></i><sup>2</sup> &minus; <i>w</i><sub><i>j,n</i>&minus;1</sub> <i>w</i><sub><i>j,n</i>+1</sub>.</span>
  <span>(S5)</span>
</div>

<p><b>6) Cox&ndash;de Boor Cubic B-Spline Recursion in KAN:</b> On a uniform knot sequence {<i>t</i><sub>0</sub>, <i>t</i><sub>1</sub>, &hellip;, <i>t</i><sub><i>G</i>+2<i>k</i></sub>} over [&minus;2.5, 2.5], the degree-<i>k</i> (<i>k</i> = 3) B-spline basis functions <i>B<sub>m,k</sub></i>(<i>z</i>) used in Eq. (2) of the main manuscript are evaluated recursively from piecewise-constant indicators:</p>
<div class="eq">
  <span><i>B</i><sub><i>m</i>,0</sub>(<i>z</i>) = I(<i>t<sub>m</sub></i> &le; <i>z</i> &lt; <i>t</i><sub><i>m</i>+1</sub>), &emsp; <i>B<sub>m,k</sub></i>(<i>z</i>) = [(<i>z</i> &minus; <i>t<sub>m</sub></i>)/(<i>t</i><sub><i>m</i>+<i>k</i></sub> &minus; <i>t<sub>m</sub></i>)] <i>B</i><sub><i>m,k</i>&minus;1</sub>(<i>z</i>) + [(<i>t</i><sub><i>m</i>+<i>k</i>+1</sub> &minus; <i>z</i>)/(<i>t</i><sub><i>m</i>+<i>k</i>+1</sub> &minus; <i>t</i><sub><i>m</i>+1</sub>)] <i>B</i><sub><i>m</i>+1,<i>k</i>&minus;1</sub>(<i>z</i>).</span>
  <span>(S6)</span>
</div>

<h2>Section S2. Hyperparameter Tuning &amp; Architecture Specifications for All 13 Models</h2>
<p>Table S1 details the exact architectural configurations, hyperparameter search spaces, and training settings for all 13 models compared in Section IV-B and Fig. 3 of the main manuscript. All models were evaluated on the identical five-fold <code>GroupKFold</code> patient splits.</p>

<table>
  <thead>
    <tr>
      <th class="l">Model Family</th>
      <th class="l">Architecture / Classifier</th>
      <th class="l">Key Hyperparameters &amp; Regularization</th>
      <th class="l">Optimizer &amp; Stopping Criteria</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="l">Gradient Boosted Trees</td>
      <td class="l"><b>LightGBM</b> (<i>&eta;</i> = 0.05 &amp; <i>&eta;</i> = 0.10)</td>
      <td class="l"><code>n_estimators=100</code>, <code>num_leaves=31</code>, <code>max_depth=-1</code>, <code>subsample=0.85</code>, <code>colsample_bytree=0.85</code></td>
      <td class="l">Exact leaf-wise greedy split; cross-entropy objective</td>
    </tr>
    <tr>
      <td class="l">Gradient Boosted Trees</td>
      <td class="l"><b>XGBoost</b></td>
      <td class="l"><code>n_estimators=100</code>, <code>max_depth=6</code>, <code>learning_rate=0.10</code>, <code>subsample=0.85</code>, <code>colsample_bytree=0.85</code></td>
      <td class="l">Histogram depth-wise tree boosting; log-loss objective</td>
    </tr>
    <tr>
      <td class="l">Gradient Boosted Trees</td>
      <td class="l"><b>CatBoost</b></td>
      <td class="l"><code>iterations=100</code>, <code>depth=6</code>, <code>learning_rate=0.08</code>, <code>l2_leaf_reg=3.0</code> (oblivious trees)</td>
      <td class="l">Symmetric tree splitting; binary log-loss</td>
    </tr>
    <tr>
      <td class="l">Spline Network</td>
      <td class="l"><b>KAN</b> (Cubic B-Spline)</td>
      <td class="l"><code>width=[D_in, 8, 2]</code>, spline order <code>k=3</code>, grid intervals <code>G=5</code>, SiLU residual base, <i>&lambda;</i><sub>L1</sub> = 10<sup>&minus;4</sup></td>
      <td class="l">Adam (lr = 0.01, weight decay 10<sup>&minus;3</sup>), 60 epochs</td>
    </tr>
    <tr>
      <td class="l">Classical Non-Linear</td>
      <td class="l"><b>MLP</b> (2-Layer Perceptron)</td>
      <td class="l"><code>hidden_layer_sizes=(64, 32)</code>, ReLU activation, <i>L</i><sub>2</sub> penalty <i>&alpha;</i> = 10<sup>&minus;4</sup>, batch size 64</td>
      <td class="l">Adam (lr = 0.001), max 150 epochs</td>
    </tr>
    <tr>
      <td class="l">Kernel Classifier</td>
      <td class="l"><b>SVM</b> (RBF Kernel)</td>
      <td class="l">Radial Basis Function kernel, regularization <i>C</i> = 1.0, kernel scale <i>&gamma;</i> = 1/(<i>D</i><sub>in</sub><i>&sigma;<sub>X</sub></i><sup>2</sup>)</td>
      <td class="l">SMO solver + 5-fold internal Platt scaling for probabilities</td>
    </tr>
    <tr>
      <td class="l">Deep Tabular Attention</td>
      <td class="l"><b>FT-Transformer</b></td>
      <td class="l">Feature Tokenizer + <code>num_attn_blocks=2</code>, <code>num_heads=2</code>, <code>input_embed_dim=32</code>, dropout 0.1</td>
      <td class="l">AdamW (lr = 10<sup>&minus;3</sup>), batch size 128, early stopping (patience 3)</td>
    </tr>
    <tr>
      <td class="l">Deep Tabular Attention</td>
      <td class="l"><b>SAINT</b> (Row + Col Attention)</td>
      <td class="l">Intersample + intrasample self-attention, <code>num_attn_blocks=2</code>, <code>num_heads=2</code>, dropout 0.1</td>
      <td class="l">AdamW (lr = 10<sup>&minus;3</sup>), batch size 128, early stopping (patience 3)</td>
    </tr>
    <tr>
      <td class="l">Deep Tabular Attention</td>
      <td class="l"><b>TabTransformer</b></td>
      <td class="l">Contextual transformer blocks (<code>num_attn_blocks=2</code>, <code>num_heads=2</code>) + MLP head <code>[64, 32]</code></td>
      <td class="l">AdamW (lr = 10<sup>&minus;3</sup>), batch size 128, early stopping (patience 3)</td>
    </tr>
    <tr>
      <td class="l">Sparse Attention</td>
      <td class="l"><b>TabNet</b></td>
      <td class="l">Sequential Sparsemax feature masks, <code>n_d=8</code>, <code>n_a=8</code>, <code>n_steps=3</code>, <i>&gamma;</i> = 1.3, <i>&lambda;</i><sub>sparse</sub> = 10<sup>&minus;3</sup></td>
      <td class="l">Adam (lr = 0.02), batch size 128, early stopping (patience 3)</td>
    </tr>
    <tr>
      <td class="l">Differentiable Trees</td>
      <td class="l"><b>NODE</b> &amp; <b>GATE</b></td>
      <td class="l">NODE: <code>num_layers=1</code>, <code>num_trees=64</code>, <code>depth=4</code>; GATE: <code>gflu_stages=2</code>, <code>num_trees=10</code>, <code>tree_depth=4</code></td>
      <td class="l">AdamW (lr = 10<sup>&minus;3</sup>), entmax split relaxation</td>
    </tr>
  </tbody>
</table>

<h2>Section S3. Cross-Corpus 27-Feature and 13-Feature Homologous Acoustic Mapping</h2>
<p>Table S2 lists all 27 homologous acoustic biomarkers aligned between the 753-feature Turkish UCI sustained vowel corpus (<code>TR</code>) and the 71-feature Bengali BenSParX conversational corpus (<code>BN</code>), along with physical differences in signal acquisition and normalization. For the 22-feature English Oxford corpus (<code>OX</code>), the 13 shared dysphonia features mapped to TR are: <code>MDVP:Jitter(%)</code> &harr; <code>locPctJitter</code>, <code>MDVP:Jitter(Abs)</code> &harr; <code>locAbsJitter</code>, <code>MDVP:RAP</code> &harr; <code>rapJitter</code>, <code>MDVP:PPQ</code> &harr; <code>ppq5Jitter</code>, <code>Jitter:DDP</code> &harr; <code>ddpJitter</code>, <code>MDVP:Shimmer</code> &harr; <code>locShimmer</code>, <code>MDVP:Shimmer(dB)</code> &harr; <code>locDbShimmer</code>, <code>Shimmer:APQ3</code> &harr; <code>apq3Shimmer</code>, <code>Shimmer:APQ5</code> &harr; <code>apq5Shimmer</code>, <code>MDVP:APQ</code> &harr; <code>apq11Shimmer</code>, <code>Shimmer:DDA</code> &harr; <code>ddaShimmer</code>, <code>NHR</code> &harr; <code>meanNoiseToHarmHarmonicity</code>, and <code>HNR</code> &harr; <code>meanHarmToNoiseHarmonicity</code>.</p>

<table>
  <thead>
    <tr>
      <th>#</th>
      <th class="l">Acoustic Feature Family</th>
      <th class="l">Turkish UCI Column (TR)</th>
      <th class="l">Bengali BenSParX Column (BN)</th>
      <th class="l">Physical Task &amp; Extraction Protocol Difference</th>
    </tr>
  </thead>
  <tbody>
    __MAP_TABLE_HTML__
  </tbody>
</table>

<p><b>Table S3 (Static vs. Dynamic MFCC Correspondence and Polarity Analysis):</b> Beyond the 27 shared perturbation, formant, and energy features in Table S2, both TR and BN extract 13 static Mel-Frequency Cepstral Coefficients (<code>mean_MFCC_0th_coef</code>&ndash;<code>mean_MFCC_12th_coef</code> in TR vs. <code>MFCC_1</code>&ndash;<code>MFCC_13</code> in BN) and their first-order temporal derivatives (<code>mean_1st_delta</code>&ndash;<code>mean_12th_delta</code> in TR vs. <code>Delta_MFCC_1</code>&ndash;<code>Delta_MFCC_12</code> in BN). As shown in Section IV-D and Fig. 5(c) of the main manuscript, static spectral envelope coefficients such as <code>MFCC_2</code> preserve consistent positive effect directionality across languages (TR Cohen's <i>d</i> = +0.95; BN Cohen's <i>d</i> = +0.96), whereas first-order cepstral velocities (<code>Delta_MFCC_2</code>) invert sign between sustained vowel phonation (<i>d</i> = &minus;0.24 in TR) and spontaneous conversation (<i>d</i> = +0.96 in BN) due to stationary vocal tract rigidity versus dynamic articulatory transition undershoot.</p>

<h2 style="page-break-before: always;">Section S4. Supplementary Confusion Matrices and Training Convergence Curves</h2>
<div class="fig">
  <img src="{fig_dir}/FigS1_Confusion_Matrices.png" alt="Supplementary Fig. S1 Confusion Matrices">
  <div class="cap"><b>Supplementary Fig. S1.</b> Out-of-fold patient-wise confusion matrices comparing unadjusted threshold classification (<i>&tau;</i> = 0.50, top row) against inner-fold Youden's <i>&tau;</i><sup>*</sup> threshold calibration and subject-level voting (bottom row) across Turkish (TR), Bengali (BN), and English Oxford (OX) cohorts.</div>
</div>

<div class="fig">
  <img src="{fig_dir}/FigS2_Training_Convergence.png" alt="Supplementary Fig. S2 Training Convergence">
  <div class="cap"><b>Supplementary Fig. S2.</b> Training dynamics across 60 epochs on the Turkish (TR) cohort: <b>(a)</b> Cross-entropy training loss convergence comparing KAN (<code>[753, 8, 2]</code>), MLP (<code>[753, 64, 32, 2]</code>), and FT-Transformer; <b>(b)</b> Out-of-fold patient-wise ROC-AUC progression across training epochs.</div>
</div>

</body>
</html>
"""
    supp_html = supp_html.replace("{fig_dir}", fig_dir).replace("__MAP_TABLE_HTML__", map_table_html)
    supp_html = convert_inline_code_and_math(supp_html)
    with open(SUPP_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(supp_html)
    print(f"Saved Supplementary HTML to {SUPP_HTML_PATH}")


def compile_pdf(html_path, pdf_path):
    cmd = [
        CHROME_EXE,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        f"file:///{html_path.replace(chr(92), '/')}"
    ]
    subprocess.run(cmd, check=True)
    print(f"Compiled PDF: {pdf_path}")


def get_pdf_page_count(pdf_path):
    with open(pdf_path, "rb") as f:
        data = f.read()
    pages = re.findall(rb"/Type\s*/Page\b", data)
    return len(pages)


def check_banned_words(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        text = re.sub(r"<[^>]+>", " ", f.read())
    banned = ["prove", "proven", "catastrophic", "transformative", "synergistically", "synergistic", "paradigm", "delve", "tapestry"]
    found = {}
    for w in banned:
        matches = re.findall(rf"\b{w}\b", text, flags=re.IGNORECASE)
        if matches:
            found[w] = len(matches)
    return found


def generate_synchronized_latex_and_md(refs):
    md_text = """# Patient-Grouped Parkinson's Disease Screening Across Languages via LightGBM and Kolmogorov–Arnold Spline Ensembles

**Author:** Arunavo, *Member, IEEE*  
**Code & Data Repository:** https://github.com/Arunavo-build/Perkinson-Disease  
**Main Manuscript PDF (8 Pages):** [IEEE_Q1_Journal_Manuscript.pdf](file:///d:/Perkinson%20disease/IEEE_Q1_Journal_Manuscript.pdf)  
**Companion Supplementary Material PDF:** [IEEE_Q1_Supplementary_Material.pdf](file:///d:/Perkinson%20disease/IEEE_Q1_Supplementary_Material.pdf)

## Abstract (241 Words)
Voice recordings offer a practical, non-invasive way to screen for Parkinson's disease (PD), yet many published studies split multiple recordings from the same speaker across training and test sets. This record-wise evaluation allows classifiers to memorize speaker identity rather than vocal pathology, inflating reported accuracy by 6.6% to 16.8% on standard benchmarks. Furthermore, acoustic features extracted from sustained vowel phonation and spontaneous conversation differ in both dimensionality and physical meaning, complicating deployment across languages. In this study, we evaluate a patient-grouped classification framework across three cohorts: Turkish sustained vowels (TR, 756 recordings from 252 subjects), Bengali spontaneous speech (BN, 900 recordings from 120 subjects), and English sustained vowels (OX, 195 recordings from 32 subjects). To combine sharp decision boundaries on perturbation thresholds with smooth modeling of continuous spectral trajectories, we average probability estimates from gradient-boosted decision trees (LightGBM) and cubic B-spline Kolmogorov–Arnold Networks (KAN). Across five random seeds under strict five-fold patient-wise cross-validation, the combined model achieves recording-level accuracies of 85.08% on TR, 93.49% on BN, and 76.82% on OX, rising to 86.27%, 100.00%, and 81.25% when aggregating repeated recordings per subject. Calibrating decision thresholds inside each training fold increases specificity on imbalanced cohorts from roughly 51% to over 71%. Finally, few-shot cross-lingual experiments show that while training directly on abundant target-language data remains superior, transferring a source model aligned via Correlation Alignment (CORAL) improves screening accuracy by 5.5% to 13.0% when only 5% to 10% of target patients are available.

## Summary of Revisions Addressing Supervisor Feedback
1. **Humanized, Measured Prose & Concise Abstract (241 Words):** Rewrote the entire manuscript in clear, direct academic language. Removed all AI buzzwords (`prove`, `catastrophic`, `transformative`, `synergistically`, `paradigm`) and trimmed the abstract from ~400 words to 241 words.
2. **8-Page Main Manuscript + Dedicated Supplementary Material:** Moved standard acoustic equations (Jitter, Shimmer, HNR, PPE, MFCCs, TQWT, Cox–de Boor recursion), 13-model hyperparameter tuning tables, the 27-feature homologous mapping table, confusion matrices (`Fig. S1`), and training convergence curves (`Fig. S2`) to `IEEE_Q1_Supplementary_Material.pdf`.
3. **More Graphical Figures (7) than Tables (3):** Replaced monotonous results/ablation tables with graphical multi-panel plots (`Figs. 1–7`), including a compact 4-panel graphical system overview (`Fig. 1`) replacing the old flowchart.
4. **Multi-Seed Evaluation (`[42, 100, 2024, 777, 999]`), Ensemble Partner Ablations, Subject-Level Voting, & Patient-Clustered Statistical Tests:** Evaluated all models across 5 seeds; added `LightGBM + MLP`, `LightGBM + SVM`, and `Logistic Stacking` ablations; reported Obuchowski's patient-clustered McNemar test ($S=252, 120, 32$) and 25-fold Wilcoxon tests; corrected the `MFCC_2` vs. `Delta_MFCC_2` polarity analysis; and reframed cross-lingual transfer with 0%–50% target patient learning curves (`Fig. 7`).
"""
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_text)
    shutil.copyfile(MD_PATH, ARTIFACT_MD_PATH)


def main():
    fig_dir = "file:///" + os.path.join(WORKSPACE_DIR, "figures").replace("\\", "/")
    refs = build_references_list()
    generate_main_html(fig_dir, refs)
    generate_supplementary_html(fig_dir, refs)

    compile_pdf(HTML_PATH, PDF_PATH)
    compile_pdf(SUPP_HTML_PATH, SUPP_PDF_PATH)

    shutil.copyfile(PDF_PATH, PDF_ALIAS_PATH)
    shutil.copyfile(PDF_PATH, PAPER_PDF_PATH)
    shutil.copyfile(PDF_PATH, ARTIFACT_PDF_PATH)
    shutil.copyfile(SUPP_PDF_PATH, ARTIFACT_SUPP_PDF_PATH)

    generate_synchronized_latex_and_md(refs)

    main_pages = get_pdf_page_count(PDF_PATH)
    supp_pages = get_pdf_page_count(SUPP_PDF_PATH)
    banned_found = check_banned_words(HTML_PATH)
    print(f"Main Manuscript Page Count: {main_pages} pages (Target <= 8 pages)")
    print(f"Supplementary Material Page Count: {supp_pages} pages")
    print(f"Banned Buzzwords Check: {banned_found}")


if __name__ == "__main__":
    main()
