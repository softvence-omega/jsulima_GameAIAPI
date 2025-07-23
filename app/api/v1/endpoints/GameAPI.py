from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_game_info():
    return {"game": "This is the GameAPI endpoint."}