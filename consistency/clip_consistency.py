from utils import CLIPConsistency

checker = CLIPConsistency()

def get_response(image_path, caption, client=None):
    score = checker.get_similarity(image_path, caption)
    
    verdict = "TRUE" if score >= 0.28 else "FALSE"
    
    return f"<verdict>{verdict}</verdict> <score>{score}</score>"