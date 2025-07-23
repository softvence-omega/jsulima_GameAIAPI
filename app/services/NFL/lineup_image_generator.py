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
        self.width, self.height = 1000, 600
        self.image = Image.new("RGB", (self.width, self.height), "#3A7D3A")
        self.draw = ImageDraw.Draw(self.image, 'RGBA')
        try:
            self.font_player = ImageFont.truetype("arialbd.ttf", 12)
            self.font_jersey = ImageFont.truetype("arialbd.ttf", 24)
            self.font_pos = ImageFont.truetype("arial.ttf", 12)
            self.font_yard_line = ImageFont.truetype("arialbd.ttf", 24)
        except IOError:
            self.font_player = ImageFont.load_default()
            self.font_jersey = ImageFont.load_default()
            self.font_pos = ImageFont.load_default()
            self.font_yard_line = ImageFont.load_default()

    def _draw_field(self):
        chalk_white = (255, 255, 255, 230)
        line_width = 2
        
        # Draw yard lines and numbers
        for yards in range(10, 50, 10):
            # From left to center
            x_left = self.width * (yards / 100.0)
            self.draw.line([(x_left, 0), (x_left, self.height)], fill=chalk_white, width=line_width)
            self.draw.text((x_left + 5, 10), str(yards), font=self.font_yard_line, fill=chalk_white)
            self.draw.text((x_left + 5, self.height - 40), str(yards), font=self.font_yard_line, fill=chalk_white)

            # From right to center
            x_right = self.width * ((100 - yards) / 100.0)
            self.draw.line([(x_right, 0), (x_right, self.height)], fill=chalk_white, width=line_width)
            self.draw.text((x_right - 30, 10), str(yards), font=self.font_yard_line, fill=chalk_white)
            self.draw.text((x_right - 30, self.height - 40), str(yards), font=self.font_yard_line, fill=chalk_white)

        # 50-yard line
        fifty_x = self.width / 2
        self.draw.line([(fifty_x, 0), (fifty_x, self.height)], fill=chalk_white, width=3)
        self.draw.text((fifty_x - 15, 10), "50", font=self.font_yard_line, fill=chalk_white)
        self.draw.text((fifty_x - 15, self.height - 40), "50", font=self.font_yard_line, fill=chalk_white)
        
        # Hash marks
        for i in range(1, 100):
             if i % 5 != 0:
                x = self.width * (i / 100.0)
                self.draw.line([(x, self.height/3), (x, self.height/3+15)], fill=chalk_white, width=1)
                self.draw.line([(x, 2*self.height/3-15), (x, 2*self.height/3)], fill=chalk_white, width=1)


    def _get_position_coords(self, is_home):
        side_multiplier = 1 if is_home else -1
        x_offset = self.width / 2 - side_multiplier * (self.width * 0.1) 
        y_center = self.height / 2
        
        ol_y_spacing = 45
        wr_y_spread = self.height * 0.4

        offense_coords = {
            "C1":   (x_offset, y_center),
            "G1":  (x_offset, y_center - ol_y_spacing),
            "G2":  (x_offset, y_center + ol_y_spacing),
            "OT1":  (x_offset, y_center - ol_y_spacing * 2),
            "OT2":  (x_offset, y_center + ol_y_spacing * 2),
            "QB1": (x_offset - side_multiplier * 25, y_center),
            "RB1": (x_offset - side_multiplier * 35, y_center + 50),
            "TE1": (x_offset, y_center + ol_y_spacing * 3),
            "WR1": (x_offset - side_multiplier * 10, y_center - wr_y_spread),
            "WR2": (x_offset - side_multiplier * 10, y_center + wr_y_spread),
            "WR3": (x_offset - side_multiplier * 10, y_center - wr_y_spread/2.5), 
        }
        
        dl_y_spacing = 50
        lb_depth = 35
        cb_depth = 70
        s_depth = 100
        
        defense_coords = {
            "DT1": (x_offset + side_multiplier * 10, y_center - dl_y_spacing / 2),
            "DT2": (x_offset + side_multiplier * 10, y_center + dl_y_spacing / 2),
            "DE1": (x_offset + side_multiplier * 10, y_center - dl_y_spacing * 2.2),
            "DE2": (x_offset + side_multiplier * 10, y_center + dl_y_spacing * 2.2),
            "LB1": (x_offset + side_multiplier * lb_depth, y_center),
            "LB2": (x_offset + side_multiplier * lb_depth, y_center - 80),
            "LB3": (x_offset + side_multiplier * lb_depth, y_center + 80),
            "CB1": (x_offset + side_multiplier * cb_depth, y_center - wr_y_spread),
            "CB2": (x_offset + side_multiplier * cb_depth, y_center + wr_y_spread),
            "S1":  (x_offset + side_multiplier * s_depth, y_center - 60),
            "S2":  (x_offset + side_multiplier * s_depth, y_center + 60),
        }
        return offense_coords, defense_coords


    def _draw_player(self, x, y, player_info, color):
        if player_info is None: return
        
        jersey_num = str(int(player_info.get('player_number', ''))) if pd.notna(player_info.get('player_number')) else ""
        # player_name = player_info.get('player_name', 'Unknown')
        pos = player_info.get('player_position', '')

        self.draw.ellipse((x - 25, y - 25, x + 25, y + 25), fill=color, outline="black", width=2)
        self.draw.text((x, y), jersey_num, font=self.font_jersey, fill="yellow", anchor="mm", stroke_width=2, stroke_fill="black")
        #self.draw.text((x, y + 35), player_name, font=self.font_player, fill="white", anchor="ms")
        self.draw.text((x, y - 28), pos, font=self.font_pos, fill="white", anchor="mb")

    def _draw_team_on_field(self, team_df, is_home, offense_coords, defense_coords):
        home_color, away_color = "#C8102E", "#00338D"
        color = home_color if is_home else away_color

        pos_counters = {}

        for _, player in team_df.iterrows():
            pos = player['player_position']
            unit = player['unit_group']
            
            pos_counters[pos] = pos_counters.get(pos, 0) + 1
            pos_key = f"{pos}{pos_counters[pos]}"
            
            coords_map = offense_coords if unit == 'offense' else defense_coords
            
            if pos_key in coords_map:
                x, y = coords_map[pos_key]
                self._draw_player(x, y, player.to_dict(), color)

    def _draw_teams(self):
        home_offense_coords, home_defense_coords = self._get_position_coords(is_home=True)
        away_offense_coords, away_defense_coords = self._get_position_coords(is_home=False)
        
        # For visualization, let's assume home is on offense and away is on defense
        self._draw_team_on_field(self.home_team_df, True, home_offense_coords, home_defense_coords)
        self._draw_team_on_field(self.away_team_df, False, away_offense_coords, away_defense_coords)


    def generate_image_base64(self):
        self._draw_field()
        self._draw_teams()
        
        buffered = io.BytesIO()
        self.image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8") 