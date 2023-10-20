class Node:
    def __init__(self, id, latitude, longitude, priority, zone_type, congestion, transfer_opp):
        self.id = id                        # Unique identifier for the node
        self.latitude = latitude            # Latitude coordinate
        self.longitude = longitude          # Longitude coordinate
        self.priority = priority            # 3=High, 2=Mid, 1=Low
        self.zone_type = zone_type          # 1=Commercial, 2=Residential
        self.congestion = congestion        # 3=High, 2=Mid, 1=Low
        self.transfer_opp = transfer_opp    # Numerical value

    def __repr__(self):
        return f"Node(id={self.id}, latitude={self.latitude}, longitude={self.longitude}, priority_level={self.priority}, zone_type={self.zone_type}, congestion_level={self.congestion}, transfer_opportunities={self.transfer_opp})"
