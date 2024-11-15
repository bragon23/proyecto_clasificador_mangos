import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import entropy

def get_texture_features(image):
    """
    Extrae múltiples características de textura de una imagen.
    
    Parámetros:
    image: numpy.ndarray
        Imagen en escala de grises
        
    Retorna:
    dict: Diccionario con todas las características de textura
    """
    # Asegurarse de que la imagen esté en escala de grises
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features = {}
    
    # 1. Características de Haralick (GLCM)
    features.update(get_glcm_features(image))
    
    # 2. Patrones Binarios Locales (LBP)
    features.update(get_lbp_features(image))
    
    # 3. Características estadísticas
    features.update(get_statistical_features(image))
    
    return features

def get_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Calcula características basadas en la Matriz de Co-ocurrencia de Niveles de Gris (GLCM).
    """
    # Normalizar imagen a 8 niveles de gris para reducir cálculos
    image_8 = (image/32).astype(np.uint8)
    
    # Calcular GLCM
    glcm = graycomatrix(image_8, distances, angles, levels=8, symmetric=True, normed=True)
    
    # Calcular propiedades de GLCM
    features = {
        'contrast': np.mean(graycoprops(glcm, 'contrast')),
        'dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
        'homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
        'energy': np.mean(graycoprops(glcm, 'energy')),
        'correlation': np.mean(graycoprops(glcm, 'correlation')),
        'ASM': np.mean(graycoprops(glcm, 'ASM'))
    }
    
    return features

def get_lbp_features(image, n_points=8, radius=1):
    """
    Calcula características basadas en Patrones Binarios Locales (LBP).
    """
    # Calcular LBP
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    # Calcular histograma de LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    # Calcular estadísticas del LBP
    features = {
        'lbp_entropy': entropy(hist),
        'lbp_energy': np.sum(hist**2),
        'lbp_mean': np.mean(lbp),
        'lbp_variance': np.var(lbp)
    }
    
    return features

def get_statistical_features(image):
    """
    Calcula características estadísticas de la textura.
    """
    # Calcular gradientes
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    features = {
        # Estadísticas básicas
        'mean': np.mean(image),
        'std': np.std(image),
        'skewness': np.mean(((image - np.mean(image))/np.std(image))**3),
        'kurtosis': np.mean(((image - np.mean(image))/np.std(image))**4) - 3,
        
        # Estadísticas de gradiente
        'gradient_mean': np.mean(gradient_magnitude),
        'gradient_std': np.std(gradient_magnitude),
        
        # Entropía
        'entropy': entropy(image.ravel())
    }
    
    return features

def analyze_texture_regions(image, region_size=32):
    """
    Analiza la textura por regiones de la imagen.
    
    Útil para detectar variaciones locales en la textura que podrían indicar 
    áreas deterioradas en el mango.
    """
    height, width = image.shape
    features_map = []
    
    for y in range(0, height, region_size):
        row = []
        for x in range(0, width, region_size):
            # Extraer región
            region = image[y:min(y+region_size, height), 
                         x:min(x+region_size, width)]
            
            # Obtener características de la región
            region_features = get_texture_features(region)
            row.append(region_features)
        features_map.append(row)
    
    return features_map
