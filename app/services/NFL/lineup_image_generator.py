# from PIL import Image, ImageDraw, ImageFont
# import io
# import base64
# import pandas as pd

# class NFLTeamLineupImageGenerator:
#     """
#     Generates a stylized NFL lineup image showing home and away teams on a football field.
#     """
#     def __init__(self, home_team_df, away_team_df):
#         self.home_team_df = home_team_df
#         self.away_team_df = away_team_df
#         self.width, self.height = 1000, 600
#         self.image = Image.new("RGB", (self.width, self.height), "#3A7D3A")
#         self.draw = ImageDraw.Draw(self.image, 'RGBA')
#         try:
#             self.font_player = ImageFont.truetype("arialbd.ttf", 12)
#             self.font_jersey = ImageFont.truetype("arialbd.ttf", 24)
#             self.font_pos = ImageFont.truetype("arial.ttf", 12)
#             self.font_yard_line = ImageFont.truetype("arialbd.ttf", 24)
#         except IOError:
#             self.font_player = ImageFont.load_default()
#             self.font_jersey = ImageFont.load_default()
#             self.font_pos = ImageFont.load_default()
#             self.font_yard_line = ImageFont.load_default()

#     def _draw_field(self):
#         chalk_white = (255, 255, 255, 230)
#         line_width = 2
        
#         # Draw yard lines and numbers
#         for yards in range(10, 50, 10):
#             # From left to center
#             x_left = self.width * (yards / 100.0)
#             self.draw.line([(x_left, 0), (x_left, self.height)], fill=chalk_white, width=line_width)
#             self.draw.text((x_left + 5, 10), str(yards), font=self.font_yard_line, fill=chalk_white)
#             self.draw.text((x_left + 5, self.height - 40), str(yards), font=self.font_yard_line, fill=chalk_white)

#             # From right to center
#             x_right = self.width * ((100 - yards) / 100.0)
#             self.draw.line([(x_right, 0), (x_right, self.height)], fill=chalk_white, width=line_width)
#             self.draw.text((x_right - 30, 10), str(yards), font=self.font_yard_line, fill=chalk_white)
#             self.draw.text((x_right - 30, self.height - 40), str(yards), font=self.font_yard_line, fill=chalk_white)

#         # 50-yard line
#         fifty_x = self.width / 2
#         self.draw.line([(fifty_x, 0), (fifty_x, self.height)], fill=chalk_white, width=3)
#         self.draw.text((fifty_x - 15, 10), "50", font=self.font_yard_line, fill=chalk_white)
#         self.draw.text((fifty_x - 15, self.height - 40), "50", font=self.font_yard_line, fill=chalk_white)
        
#         # Hash marks
#         for i in range(1, 100):
#              if i % 5 != 0:
#                 x = self.width * (i / 100.0)
#                 self.draw.line([(x, self.height/3), (x, self.height/3+15)], fill=chalk_white, width=1)
#                 self.draw.line([(x, 2*self.height/3-15), (x, 2*self.height/3)], fill=chalk_white, width=1)


#     def _get_position_coords(self, is_home):
#         side_multiplier = 1 if is_home else -1
#         x_offset = self.width / 2 - side_multiplier * (self.width * 0.1) 
#         y_center = self.height / 2
        
#         ol_y_spacing = 45
#         wr_y_spread = self.height * 0.4

#         offense_coords = {
#             "C1":   (x_offset, y_center),
#             "G1":  (x_offset, y_center - ol_y_spacing),
#             "G2":  (x_offset, y_center + ol_y_spacing),
#             "OT1":  (x_offset, y_center - ol_y_spacing * 2),
#             "OT2":  (x_offset, y_center + ol_y_spacing * 2),
#             "QB1": (x_offset - side_multiplier * 25, y_center),
#             "RB1": (x_offset - side_multiplier * 35, y_center + 50),
#             "TE1": (x_offset, y_center + ol_y_spacing * 3),
#             "WR1": (x_offset - side_multiplier * 10, y_center - wr_y_spread),
#             "WR2": (x_offset - side_multiplier * 10, y_center + wr_y_spread),
#             "WR3": (x_offset - side_multiplier * 10, y_center - wr_y_spread/2.5), 
#         }
        
#         dl_y_spacing = 50
#         lb_depth = 35
#         cb_depth = 70
#         s_depth = 100
        
#         defense_coords = {
#             "DT1": (x_offset + side_multiplier * 10, y_center - dl_y_spacing / 2),
#             "DT2": (x_offset + side_multiplier * 10, y_center + dl_y_spacing / 2),
#             "DE1": (x_offset + side_multiplier * 10, y_center - dl_y_spacing * 2.2),
#             "DE2": (x_offset + side_multiplier * 10, y_center + dl_y_spacing * 2.2),
#             "LB1": (x_offset + side_multiplier * lb_depth, y_center),
#             "LB2": (x_offset + side_multiplier * lb_depth, y_center - 80),
#             "LB3": (x_offset + side_multiplier * lb_depth, y_center + 80),
#             "CB1": (x_offset + side_multiplier * cb_depth, y_center - wr_y_spread),
#             "CB2": (x_offset + side_multiplier * cb_depth, y_center + wr_y_spread),
#             "S1":  (x_offset + side_multiplier * s_depth, y_center - 60),
#             "S2":  (x_offset + side_multiplier * s_depth, y_center + 60),
#         }
#         return offense_coords, defense_coords


#     def _draw_player(self, x, y, player_info, color):
#         if player_info is None: return
        
#         jersey_num = str(int(player_info.get('player_number', ''))) if pd.notna(player_info.get('player_number')) else ""
#         # player_name = player_info.get('player_name', 'Unknown')
#         pos = player_info.get('player_position', '')

#         self.draw.ellipse((x - 25, y - 25, x + 25, y + 25), fill=color, outline="black", width=2)
#         self.draw.text((x, y), jersey_num, font=self.font_jersey, fill="yellow", anchor="mm", stroke_width=2, stroke_fill="black")
#         #self.draw.text((x, y + 35), player_name, font=self.font_player, fill="white", anchor="ms")
#         self.draw.text((x, y - 28), pos, font=self.font_pos, fill="white", anchor="mb")

#     def _draw_team_on_field(self, team_df, is_home, offense_coords, defense_coords):
#         home_color, away_color = "#C8102E", "#00338D"
#         color = home_color if is_home else away_color

#         pos_counters = {}

#         for _, player in team_df.iterrows():
#             pos = player['player_position']
#             unit = "offense" #player['unit_group']
            
#             pos_counters[pos] = pos_counters.get(pos, 0) + 1
#             pos_key = f"{pos}{pos_counters[pos]}"
            
#             coords_map = offense_coords if unit == 'offense' else defense_coords
            
#             if pos_key in coords_map:
#                 x, y = coords_map[pos_key]
#                 self._draw_player(x, y, player.to_dict(), color)

#     def _draw_teams(self):
#         home_offense_coords, home_defense_coords = self._get_position_coords(is_home=True)
#         away_offense_coords, away_defense_coords = self._get_position_coords(is_home=False)
        
#         # For visualization, let's assume home is on offense and away is on defense
#         self._draw_team_on_field(self.home_team_df, True, home_offense_coords, home_defense_coords)
#         self._draw_team_on_field(self.away_team_df, False, away_offense_coords, away_defense_coords)


#     def generate_image_base64(self):
#         self._draw_field()
#         self._draw_teams()
        
#         buffered = io.BytesIO()
#         self.image.save(buffered, format="PNG")
#         return base64.b64encode(buffered.getvalue()).decode("utf-8") 

from PIL import Image, ImageDraw, ImageFont
import io
import base64
import pandas as pd

class NFLTeamLineupImageGenerator:
    """
    Generates a stylized NFL lineup image showing home and away teams on a football field.
    """
    def __init__(self, home_team_df, away_team_df):
        self.home_team_df = home_team_df
        self.away_team_df = away_team_df
        self.width, self.height = 1200, 700  # Increased size for better spacing
        self.image = Image.new("RGB", (self.width, self.height), "#2D5A2D")  # Darker field green
        self.draw = ImageDraw.Draw(self.image, 'RGBA')
        self._load_fonts()

    def _load_fonts(self):
        """Load fonts with fallbacks"""
        try:
            self.font_player = ImageFont.truetype("arialbd.ttf", 14)
            self.font_jersey = ImageFont.truetype("arialbd.ttf", 20)
            self.font_pos = ImageFont.truetype("arial.ttf", 11)
            self.font_yard_line = ImageFont.truetype("arialbd.ttf", 20)
            self.font_team = ImageFont.truetype("arialbd.ttf", 24)
        except (IOError, OSError):
            # Fallback to default font
            self.font_player = ImageFont.load_default()
            self.font_jersey = ImageFont.load_default()
            self.font_pos = ImageFont.load_default()
            self.font_yard_line = ImageFont.load_default()
            self.font_team = ImageFont.load_default()

    def _draw_field(self):
        """Draw the football field with yard lines and markings"""
        chalk_white = (255, 255, 255, 230)
        line_width = 2
        
        # Field boundaries
        self.draw.rectangle([50, 50, self.width-50, self.height-50], 
                          outline=chalk_white, width=3)
        
        # Adjust field dimensions for drawing area
        field_left = 50
        field_right = self.width - 50
        field_width = field_right - field_left
        
        # Draw yard lines and numbers
        for yards in range(10, 50, 10):
            # From left to center
            x_left = field_left + (field_width * (yards / 100.0))
            self.draw.line([(x_left, 50), (x_left, self.height-50)], 
                          fill=chalk_white, width=line_width)
            self.draw.text((x_left + 5, 60), str(yards), 
                          font=self.font_yard_line, fill=chalk_white)
            self.draw.text((x_left + 5, self.height - 90), str(yards), 
                          font=self.font_yard_line, fill=chalk_white)

            # From right to center (mirrored)
            x_right = field_left + (field_width * ((100 - yards) / 100.0))
            self.draw.line([(x_right, 50), (x_right, self.height-50)], 
                          fill=chalk_white, width=line_width)
            self.draw.text((x_right - 30, 60), str(yards), 
                          font=self.font_yard_line, fill=chalk_white)
            self.draw.text((x_right - 30, self.height - 90), str(yards), 
                          font=self.font_yard_line, fill=chalk_white)

        # 50-yard line (center)
        fifty_x = field_left + field_width / 2
        self.draw.line([(fifty_x, 50), (fifty_x, self.height-50)], 
                      fill=chalk_white, width=4)
        self.draw.text((fifty_x - 15, 60), "50", 
                      font=self.font_yard_line, fill=chalk_white)
        self.draw.text((fifty_x - 15, self.height - 90), "50", 
                      font=self.font_yard_line, fill=chalk_white)
        
        # Hash marks (every 5 yards)
        for i in range(5, 100, 5):
            if i % 10 != 0:  # Skip major yard lines
                x = field_left + (field_width * (i / 100.0))
                # Upper hash marks
                self.draw.line([(x, self.height/3), (x, self.height/3+20)], 
                              fill=chalk_white, width=2)
                # Lower hash marks
                self.draw.line([(x, 2*self.height/3-20), (x, 2*self.height/3)], 
                              fill=chalk_white, width=2)

    def _get_position_coords(self, is_home):
        """Get coordinates for different positions based on team side"""
        # Define field boundaries
        field_left = 50
        field_right = self.width - 50
        field_center_x = (field_left + field_right) / 2
        field_center_y = self.height / 2
        
        # Distance from center line
        team_offset = 120 if is_home else -120
        base_x = field_center_x + team_offset
        
        # Offensive formation coordinates
        offense_coords = {
            # Offensive Line (horizontal line)
            "C": (base_x, field_center_y),
            "LG": (base_x, field_center_y - 40),
            "RG": (base_x, field_center_y + 40),
            "LT": (base_x, field_center_y - 80),
            "RT": (base_x, field_center_y + 80),
            
            # Backfield
            "QB": (base_x - (30 if is_home else -30), field_center_y),
            "RB": (base_x - (50 if is_home else -50), field_center_y + 30),
            "FB": (base_x - (40 if is_home else -40), field_center_y),
            
            # Receivers and Tight End
            "TE": (base_x, field_center_y + 120),
            "WR": (base_x - (20 if is_home else -20), field_center_y - 150),
            "WR2": (base_x - (20 if is_home else -20), field_center_y + 150),
            "WR3": (base_x - (20 if is_home else -20), field_center_y - 75),
        }
        
        # Defensive formation coordinates
        defense_coords = {
            # Defensive Line
            "DT": (base_x + (20 if is_home else -20), field_center_y - 30),
            "DT2": (base_x + (20 if is_home else -20), field_center_y + 30),
            "DE": (base_x + (20 if is_home else -20), field_center_y - 100),
            "DE2": (base_x + (20 if is_home else -20), field_center_y + 100),
            
            # Linebackers
            "LB": (base_x + (50 if is_home else -50), field_center_y),
            "LB2": (base_x + (50 if is_home else -50), field_center_y - 60),
            "LB3": (base_x + (50 if is_home else -50), field_center_y + 60),
            
            # Secondary
            "CB": (base_x + (100 if is_home else -100), field_center_y - 150),
            "CB2": (base_x + (100 if is_home else -100), field_center_y + 150),
            "S": (base_x + (120 if is_home else -120), field_center_y - 50),
            "SS": (base_x + (120 if is_home else -120), field_center_y + 50),
            "FS": (base_x + (140 if is_home else -140), field_center_y),
        }
        
        return offense_coords, defense_coords

    def _normalize_position(self, pos):
        """Normalize position names to match coordinate keys"""
        pos = str(pos).upper().strip()
        
        # Position mapping dictionary
        position_map = {
            # Offensive positions
            'CENTER': 'C',
            'LEFT GUARD': 'LG',
            'RIGHT GUARD': 'RG',
            'LEFT TACKLE': 'LT',
            'RIGHT TACKLE': 'RT',
            'QUARTERBACK': 'QB',
            'RUNNING BACK': 'RB',
            'FULLBACK': 'FB',
            'TIGHT END': 'TE',
            'WIDE RECEIVER': 'WR',
            
            # Defensive positions
            'DEFENSIVE TACKLE': 'DT',
            'DEFENSIVE END': 'DE',
            'LINEBACKER': 'LB',
            'CORNERBACK': 'CB',
            'SAFETY': 'S',
            'STRONG SAFETY': 'SS',
            'FREE SAFETY': 'FS',
        }
        
        return position_map.get(pos, pos)

    def _draw_player(self, x, y, player_info, color, team_name=""):
        """Draw a single player on the field"""
        if player_info is None:
            return
        
        # Get player information
        jersey_num = str(int(player_info.get('player_number', ''))) if pd.notna(player_info.get('player_number')) else "?"
        player_name = str(player_info.get('player_name', 'Unknown'))
        pos = str(player_info.get('player_position', ''))
        
        # Ensure coordinates are within field bounds
        x = max(70, min(self.width - 70, x))
        y = max(70, min(self.height - 70, y))
        
        # Draw player circle
        circle_radius = 25
        self.draw.ellipse((x - circle_radius, y - circle_radius, 
                          x + circle_radius, y + circle_radius), 
                         fill=color, outline="white", width=2)
        
        # Draw jersey number
        self.draw.text((x, y), jersey_num, font=self.font_jersey, 
                      fill="white", anchor="mm", stroke_width=1, stroke_fill="black")
        
        # Draw position above player
        self.draw.text((x, y - 35), pos, font=self.font_pos, 
                      fill="white", anchor="mb", stroke_width=1, stroke_fill="black")
        
        # Draw player name below (shortened if too long)
        if len(player_name) > 12:
            player_name = player_name[:10] + "..."
        self.draw.text((x, y + 35), player_name, font=self.font_player, 
                      fill="white", anchor="mt", stroke_width=1, stroke_fill="black")

    def _determine_unit(self, position):
        """Determine if position is offense or defense"""
        offensive_positions = ['C', 'LG', 'RG', 'LT', 'RT', 'QB', 'RB', 'FB', 'TE', 'WR']
        pos_normalized = self._normalize_position(position)
        return "offense" if pos_normalized in offensive_positions else "defense"

    def _draw_team_on_field(self, team_df, is_home, offense_coords, defense_coords, team_name=""):
        """Draw all players from a team on the field"""
        # Team colors
        home_color = "#C8102E"  # Red
        away_color = "#00338D"  # Blue
        color = home_color if is_home else away_color
        
        # Count positions to handle duplicates
        pos_counters = {}
        
        for _, player in team_df.iterrows():
            pos = str(player['player_position']).strip()
            pos_normalized = self._normalize_position(pos)
            unit = self._determine_unit(pos)
            
            # Handle multiple players at same position
            if pos_normalized not in pos_counters:
                pos_counters[pos_normalized] = 1
            else:
                pos_counters[pos_normalized] += 1
            
            # Create position key for multiple players
            pos_key = pos_normalized
            if pos_counters[pos_normalized] > 1:
                pos_key = f"{pos_normalized}{pos_counters[pos_normalized]}"
            
            # Get coordinates based on unit
            coords_map = offense_coords if unit == 'offense' else defense_coords
            
            # Find coordinates for this position
            if pos_key in coords_map:
                x, y = coords_map[pos_key]
                self._draw_player(x, y, player.to_dict(), color, team_name)
            elif pos_normalized in coords_map:
                x, y = coords_map[pos_normalized]
                # Offset for multiple players at same position
                offset = (pos_counters[pos_normalized] - 1) * 30
                y += offset if pos_counters[pos_normalized] % 2 == 0 else -offset
                self._draw_player(x, y, player.to_dict(), color, team_name)

    def _draw_team_labels(self):
        """Draw team labels"""
        # Home team (right side)
        self.draw.text((self.width - 100, 20), "HOME", 
                      font=self.font_team, fill="#C8102E", anchor="mt")
        
        # Away team (left side)  
        self.draw.text((100, 20), "AWAY", 
                      font=self.font_team, fill="#00338D", anchor="mt")

    def _draw_teams(self):
        """Draw both teams on the field"""
        # Get position coordinates for both teams
        home_offense_coords, home_defense_coords = self._get_position_coords(is_home=True)
        away_offense_coords, away_defense_coords = self._get_position_coords(is_home=False)
        
        # Draw away team (left side - typically on defense)
        self._draw_team_on_field(self.away_team_df, False, away_offense_coords, away_defense_coords, "Away")
        
        # Draw home team (right side - typically on offense)
        self._draw_team_on_field(self.home_team_df, True, home_offense_coords, home_defense_coords, "Home")

    def generate_image_base64(self):
        """Generate the complete lineup image and return as base64"""
        try:
            self._draw_field()
            self._draw_teams()
            self._draw_team_labels()
            
            buffered = io.BytesIO()
            self.image.save(buffered, format="PNG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def save_image(self, filename="nfl_lineup.png"):
        """Save the image to a file"""
        try:
            self._draw_field()
            self._draw_teams()
            self._draw_team_labels()
            self.image.save(filename, "PNG", quality=95)
            print(f"Image saved as {filename}")
        except Exception as e:
            print(f"Error saving image: {e}")

# Example usage:
# generator = NFLTeamLineupImageGenerator(home_df, away_df)
# base64_image = generator.generate_image_base64()
# generator.save_image("lineup.png")