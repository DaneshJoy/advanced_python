import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from eeg_generator import load_real_eeg, filter_eeg

# New BCI functions for real data
def prepare_epochs(raw: mne.io.Raw):
    """Prepare epochs for motor imagery classification."""
    events, event_id = mne.events_from_annotations(raw)
    # Event IDs: T1=2 (left), T2=3 (right) in EEGBCI
    event_id = {'left': 2, 'right': 3}
    epochs = mne.Epochs(
        raw, events, event_id, tmin=0, tmax=4, baseline=None, preload=True
    )
    return epochs, event_id

def train_classifier(epochs, event_id):
    """Train a simple CSP + LDA classifier."""
    labels = epochs.events[:, -1]
    labels = np.where(labels == 2, 0, 1)  # 0: left, 1: right
    data = epochs.get_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    
    # CSP feature extraction
    csp = CSP(n_components=4, reg=None, log=True)
    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)
    
    # LDA classifier
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train_csp, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_csp)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classifier accuracy: {acc:.2f}")
    
    return csp, clf

def classify_and_move(epochs, csp, clf):
    """Classify epochs and simulate cube movement in Pygame."""
    labels = epochs.events[:, -1]
    labels = np.where(labels == 2, 0, 1)  # 0: left, 1: right
    data = epochs.get_data()
    X_csp = csp.transform(data)
    predictions = clf.predict(X_csp)
    
    # Initialize Pygame and OpenGL for simple 3D cube
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(75, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    
    cube_position = 0.0  # Start at center
    
    for pred in predictions:
        if pred == 0:  # Left
            cube_position -= 0.5
        else:  # Right
            cube_position += 0.5
        
        # Render cube
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glTranslatef(cube_position - cube_position_prev if 'cube_position_prev' in locals() else 0, 0, 0)
        cube_position_prev = cube_position
        draw_cube()
        pygame.display.flip()
        pygame.time.wait(500)  # Pause for visibility
    
    pygame.quit()

def draw_cube():
    """Draw a simple cube using OpenGL."""
    vertices = (
        (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
        (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
    )
    edges = (
        (0,1), (0,3), (0,4), (2,1), (2,3), (2,7),
        (6,3), (6,4), (6,7), (5,1), (5,4), (5,7)
    )
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

# Example usage for real data
if __name__ == "__main__":
    raw = load_real_eeg()
    filtered = filter_eeg(raw)
    epochs, event_id = prepare_epochs(filtered)
    csp, clf = train_classifier(epochs, event_id)
    classify_and_move(epochs, csp, clf)
