"""Tests to verify the pdfs, conditional expectations and APE are correctly calculated."""

from PdfGenerator import PdfGenerator
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

base_dir = "/home/pmannix/Stratification-DNS/tests/"


def test_pdf_1d():

    pdf = PdfGenerator(file_dir=base_dir + 'rbc', N_pts=2**8, frames=5)
    pdf.generate_pdf()       

    fB_num = pdf.fB
    fB_exact = np.ones(len(pdf.fB))
    
    assert np.allclose(fB_num,fB_exact,rtol=5e-03)


def test_pdf_2d():
    
    pdf = PdfGenerator(file_dir=base_dir + 'rbc', N_pts=2**8, frames=5)
    pdf.generate_pdf()
   
    Ibz = np.outer(pdf.b,pdf.z)
    dz = pdf.z[1] - pdf.z[0] 
    db = pdf.b[1] - pdf.b[0] 
    tpe = np.nansum(Ibz*pdf.fBZ * db * dz)

    assert abs(tpe - 1/6)/abs(tpe) < 5e-03


def test_ape_rbc():
    
    pdf = PdfGenerator(file_dir=base_dir + 'rbc', N_pts=2**8, frames=5)
    pdf.generate_pdf()
    tpe, ape = pdf.energetics()

    assert abs(ape - 1/6)/abs(ape) < 5e-03


def test_ape_ihc():
    
    pdf = PdfGenerator(file_dir=base_dir + 'ihc', N_pts=2**10, frames=5)
    pdf.generate_pdf()
    tpe, ape = pdf.energetics()

    assert abs(ape - 1.084e-02)/abs(ape) < 5e-03


def test_expectation_1d_rbc():

    pdf = PdfGenerator(file_dir=base_dir + 'rbc', N_pts=2**8, frames=5)
    pdf.generate_pdf()       
    pdf.generate_expectation(testing=True)

    db = pdf.b[1] - pdf.b[0]
    e_B = np.nansum( pdf.Expectations[r'\|\nabla B\|^2']['1D']['b'] * db )

    assert abs(e_B - 1) < 5e-03


def test_expectation_1d_ihc():
    
    pdf = PdfGenerator(file_dir=base_dir + 'ihc', N_pts=2**8, frames=5)
    pdf.generate_pdf()       
    pdf.generate_expectation(testing=True)

    db = pdf.b[1] - pdf.b[0]
    e_B = np.nansum( pdf.Expectations[r'\|\nabla B\|^2']['1D']['b'] * db )
    
    assert abs(e_B - 1/12)/abs(e_B) < 1e-02


def test_expectation_2d_rbc():

    pdf = PdfGenerator(file_dir=base_dir + 'rbc', N_pts=2**8, frames=5)
    pdf.generate_pdf()       
    pdf.generate_expectation(testing=True)

    dw = pdf.w[1] - pdf.w[0]
    db = pdf.b[1] - pdf.b[0]
    e_B = np.nansum( pdf.Expectations[r'\|\nabla B\|^2']['2D']['wb'] * db * dw ) 

    assert abs(e_B - 1) < 5e-03

    dz = pdf.z[1] - pdf.z[0]
    db = pdf.b[1] - pdf.b[0]
    e_B = np.nansum( pdf.Expectations[r'\|\nabla B\|^2']['2D']['bz'] * db * dz ) 

    assert abs(e_B - 1) < 5e-03


def test_expectation_2d_ihc():
    
    pdf = PdfGenerator(file_dir=base_dir + 'ihc', N_pts=2**8, frames=5)
    pdf.generate_pdf()       
    pdf.generate_expectation(testing=True)

    dw = pdf.w[1] - pdf.w[0]
    db = pdf.b[1] - pdf.b[0]
    e_B = np.nansum( pdf.Expectations[r'\|\nabla B\|^2']['2D']['wb'] * db * dw) 
    
    assert abs(e_B - 1/12)/abs(e_B) < 1e-02

    dz = pdf.z[1] - pdf.z[0]
    db = pdf.b[1] - pdf.b[0]
    e_B = np.nansum( pdf.Expectations[r'\|\nabla B\|^2']['2D']['bz'] * db * dz) 

    assert abs(e_B - 1/12)/abs(e_B) < 1e-02
