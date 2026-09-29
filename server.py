from fastapi import FastAPI
import uvicorn
import requests

app = FastAPI(title="SIH Central Urban Intelligence Dispatcher")
ACTIVE_HAZARDS = []

@app.post("/report_hazard")
def report_hazard(payload: dict):
    hazard_type = payload.get("hazard_type")
    lat = payload.get("latitude")
    lon = payload.get("longitude")
    confidence = payload.get("confidence")

    incident = {
        "id": len(ACTIVE_HAZARDS) + 1,
        "hazard": hazard_type,
        "lat": lat,
        "lon": lon,
        "confidence": round(confidence, 2),
        "action": "TICKET_AUTO_DISPATCHED_TO_CIVIC_BODY"
    }
    ACTIVE_HAZARDS.append(incident)

    print("\n" + "="*55)
    print(f"[EMERGENCY ALERT] Obstruction Detected: {hazard_type}")
    print(f"Confidence: {confidence:.2f} | GPS: {lat}, {lon}")
    print("Automated ticket logged for Municipal / PWD teams.")
    print("="*55 + "\n")

    return {"status": "SUCCESS", "incident_id": incident["id"]}

@app.get("/reroute")
def reroute_fleet(start_lat: float, start_lon: float, end_lat: float, end_lon: float):
    osrm_endpoint = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    response = requests.get(osrm_endpoint).json()
    
    if response.get("code") == "Ok":
        coords = response["routes"][0]["geometry"]["coordinates"]
        dist = response["routes"][0]["distance"]
        duration = response["routes"][0]["duration"]
        return {
            "status": "REROUTE_OPTIMAL",
            "route_coordinates": coords,
            "total_distance_km": round(dist / 1000, 2),
            "estimated_minutes": round(duration / 60, 2),
            "hazards_avoided": len(ACTIVE_HAZARDS)
        }
    return {"status": "ERROR", "message": "Routing calculation failed"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)