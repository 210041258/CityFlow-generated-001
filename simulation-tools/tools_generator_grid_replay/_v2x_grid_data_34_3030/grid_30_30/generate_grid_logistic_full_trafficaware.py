import argparse
import os
import json
import numpy as np
from generate_json_from_grid import gridToRoadnet

# Logistic function from your analysis
def logistic(x, L=302.2328, k=0.0143, x0=101.5374):
    return L / (1 + np.exp(-k * (x - x0)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=float, help="Traffic intensity input for logistic model")
    parser.add_argument("--turn", action="store_true")
    parser.add_argument("--tlPlan", action="store_true")
    parser.add_argument("--dir", type=str, default="./")
    return parser.parse_args()

def generate_route(rowNum, colNum, turn=False):
    routes = []
    move = [(1,0),(0,1),(-1,0),(0,-1)]
    def get_straight_route(start, direction, step):
        x, y = start
        route = []
        for _ in range(step):
            route.append(f"road_{x}_{y}_{direction}")
            x += move[direction][0]
            y += move[direction][1]
        return route
    for i in range(1,rowNum+1):
        routes.append(get_straight_route((0,i),0,colNum+1))
        routes.append(get_straight_route((colNum+1,i),2,colNum+1))
    for i in range(1,colNum+1):
        routes.append(get_straight_route((i,0),1,rowNum+1))
        routes.append(get_straight_route((i,rowNum+1),3,rowNum+1))
    if turn:
        def get_turn_route(start,direction):
            if direction[0]%2==0:
                step=min(rowNum*2,colNum*2+1)
            else:
                step=min(colNum*2,rowNum*2+1)
            x,y=start
            route=[]
            cur=0
            for _ in range(step):
                route.append(f"road_{x}_{y}_{direction[cur]}")
                x+=move[direction[cur]][0]
                y+=move[direction[cur]][1]
                cur=1-cur
            return route
        routes.append(get_turn_route((1,0),(1,0)))
        routes.append(get_turn_route((0,1),(0,1)))
        routes.append(get_turn_route((colNum+1,rowNum),(2,3)))
        routes.append(get_turn_route((colNum,rowNum+1),(3,2)))
        routes.append(get_turn_route((0,rowNum),(0,3)))
        routes.append(get_turn_route((1,rowNum+1),(3,0)))
        routes.append(get_turn_route((colNum+1,1),(2,1)))
        routes.append(get_turn_route((colNum,0),(1,2)))
    return routes

if __name__=="__main__":
    args=parse_args()

    # Logistic output
    y = logistic(args.x)

    # Dynamic simulation parameters
    rowNum = max(2, int(y // 10))
    colNum = max(2, int(y // 10))

    # Traffic-aware lanes (all lane types)
    numStraightLanes = min(5, max(1, int(y // 100)))
    numLeftLanes = max(1, numStraightLanes // 2)
    numRightLanes = max(1, numStraightLanes // 2)

    # Lane max speed reduces for high-density traffic
    laneMaxSpeed = max(5.0, 25 - (y/20))

    # Interval between vehicles: shorter for higher traffic
    interval = max(0.5, 10 / (y / 50))

    print(f"Logistic Traffic Model -> rowNum={rowNum}, colNum={colNum}, interval={interval:.2f}, "
          f"Lanes(L/S/R)={numLeftLanes}/{numStraightLanes}/{numRightLanes}, laneMaxSpeed={laneMaxSpeed:.2f}")

    # Grid dictionary
    grid = {
        "rowNumber": rowNum,
        "columnNumber": colNum,
        "rowDistances":[300]*(colNum-1),
        "columnDistances":[300]*(rowNum-1),
        "outRowDistance":300,
        "outColumnDistance":300,
        "intersectionWidths":[[30]*colNum]*rowNum,
        "numLeftLanes":numLeftLanes,
        "numStraightLanes":numStraightLanes,
        "numRightLanes":numRightLanes,
        "laneMaxSpeed":laneMaxSpeed,
        "tlPlan":args.tlPlan
    }

    roadnetFile=os.path.join(args.dir,f"roadnet_{rowNum}_{colNum}{'_turn' if args.turn else ''}.json")
    flowFile=os.path.join(args.dir,f"flow_{rowNum}_{colNum}{'_turn' if args.turn else ''}.json")

    json.dump(gridToRoadnet(**grid), open(roadnetFile,"w"),indent=2)

    vehicle_template={
        "length":5.0,
        "width":2.0,
        "maxPosAcc":2.0,
        "maxNegAcc":4.5,
        "usualPosAcc":2.0,
        "usualNegAcc":4.5,
        "minGap":2.5,
        "maxSpeed":laneMaxSpeed,
        "headwayTime":1.5
    }

    routes = generate_route(rowNum,colNum,args.turn)
    flow=[]
    for route in routes:
        flow.append({
            "vehicle":vehicle_template,
            "route":route,
            "interval":interval,
            "startTime":0,
            "endTime":-1
        })
    json.dump(flow, open(flowFile,"w"),indent=2)
    print(f"Generated: {roadnetFile} & {flowFile}")