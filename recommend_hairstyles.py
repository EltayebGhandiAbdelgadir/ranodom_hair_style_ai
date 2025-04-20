def recommend_hairstyles(face_shape):
    recommendations = {
        "Oval": ["Long Waves", "Bob Cut", "Side Swept Bangs"],
        "Round": ["Layered Shag", "High Bun", "Pixie Cut"],
        "Square": ["Wavy Layers", "Curtain Bangs", "Side Part"],
        "Heart": ["Chin-Length Bob", "Soft Curls", "Fringe"],
        "Unknown": ["Try retaking the photo for better accuracy."]
    }
    return recommendations.get(face_shape, ["No recommendations available."])

