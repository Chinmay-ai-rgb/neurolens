from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime


def generate_pdf_report(features: Dict, predictions: Dict, output_path: str = None) -> str:
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/neurolens_report_{timestamp}.pdf"
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    elements.append(Paragraph("NeuroLens+ Assessment Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    elements.append(Paragraph("Risk Assessment", heading_style))
    
    risk_data = [
        ['Condition', 'Risk Score', 'Status'],
        ['Multiple Sclerosis (MS)', f"{predictions.get('MS_risk', 0):.2%}", _get_risk_status(predictions.get('MS_risk', 0))],
        ['Parkinson\'s Disease (PD)', f"{predictions.get('PD_risk', 0):.2%}", _get_risk_status(predictions.get('PD_risk', 0))],
        ['Progressive Supranuclear Palsy (PSP)', f"{predictions.get('PSP_risk', 0):.2%}", _get_risk_status(predictions.get('PSP_risk', 0))],
        ['Sixth Nerve Palsy (CN6)', f"{predictions.get('CN6_risk', 0):.2%}", _get_risk_status(predictions.get('CN6_risk', 0))]
    ]
    
    risk_table = Table(risk_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    
    elements.append(risk_table)
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph(f"Confidence: {predictions.get('confidence', 0):.2%}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    elements.append(Paragraph("Feature Metrics", heading_style))
    
    feature_categories = {
        "Fixation": ['fix_std_x', 'fix_std_y', 'fix_drift', 'fix_microsaccades', 'fix_swj', 'fix_nystagmus_hz'],
        "Saccade": ['sac_latency', 'sac_peak_vel', 'sac_amp_error', 'sac_direction_error'],
        "Smooth Pursuit": ['pursuit_gain', 'pursuit_phase_lag', 'pursuit_catchups', 'pursuit_smoothness_r2'],
        "Anti-Saccade": ['anti_error_rate', 'anti_corr_latency', 'anti_reflexive_count']
    }
    
    for category, metric_keys in feature_categories.items():
        elements.append(Paragraph(category, styles['Heading3']))
        
        metric_data = [['Metric', 'Value']]
        for key in metric_keys:
            if key in features:
                value = features[key]
                if not (isinstance(value, float) and np.isnan(value)):
                    display_name = key.replace('_', ' ').title()
                    if isinstance(value, float):
                        if 'latency' in key.lower() or 'lag' in key.lower():
                            metric_data.append([display_name, f"{value*1000:.2f} ms"])
                        elif 'rate' in key.lower():
                            metric_data.append([display_name, f"{value:.2%}"])
                        else:
                            metric_data.append([display_name, f"{value:.3f}"])
                    else:
                        metric_data.append([display_name, str(value)])
        
        if len(metric_data) > 1:
            metric_table = Table(metric_data, colWidths=[3*inch, 2*inch])
            metric_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            elements.append(metric_table)
            elements.append(Spacer(1, 0.2*inch))
    
    elements.append(PageBreak())
    
    elements.append(Paragraph("Notes", heading_style))
    elements.append(Paragraph("This report is generated from eye movement data collected during standardized neurological assessment tasks.", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("For medical interpretation, please consult with a qualified healthcare professional.", styles['Normal']))
    
    doc.build(elements)
    
    return str(output_path)


def _get_risk_status(risk: float) -> str:
    if risk > 0.7:
        return "High"
    elif risk > 0.4:
        return "Moderate"
    else:
        return "Low"

