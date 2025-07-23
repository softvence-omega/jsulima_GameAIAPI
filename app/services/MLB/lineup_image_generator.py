from PIL import Image, ImageDraw, ImageFont
import io
import base64
import pandas as pd

class LineupImageGenerator:
    """
    Generates a top-notch quality, stylized baseball lineup image,
    meticulously replicating the style of the reference image.
    This version focuses on geometric accuracy and a professional, broadcast-style aesthetic.
    """
    def __init__(self, lineup_df):
        self.lineup_df = lineup_df
        self.width, self.height = 800, 850
        self.image = Image.new("RGB", (self.width, self.height), "#1A3A1A")
        self.draw = ImageDraw.Draw(self.image, 'RGBA')
        try:
            self.font_pos = ImageFont.truetype("impact.ttf", 32)
            self.font_player = ImageFont.truetype("arialbd.ttf", 18)
            self.font_jersey = ImageFont.truetype("arialbd.ttf", 24)
        except IOError:
            self.font_pos = ImageFont.load_default()
            self.font_player = ImageFont.load_default()
            self.font_jersey = ImageFont.load_default()

    def _draw_field(self):
        # --- Define Colors ---
        grass_light = "#3C7839"
        grass_dark = "#2A5228"
        dirt = "#B08B5B"
        dirt_dark = "#9A794E"
        chalk = (255, 255, 255, 220)
        wall_color = "#102510"

        # --- Draw Field Layers ---
        self.draw.rectangle([0, 0, self.width, self.height], fill=grass_light)
        for i in range(0, self.height, 25):
            self.draw.rectangle([0, i, self.width, i + 12.5], fill=grass_dark)

        self.draw.arc([10, 10, self.width - 10, 600], 15, 165, fill=wall_color, width=35)
        
        home_x, home_y = self.width / 2, 650
        self.draw.pieslice([-150, 250, self.width + 150, 1350], 205, 335, fill=dirt)
        # self.draw.polygon([100, 400, self.width - 100, 1050], fill=grass_light)

        mound_y = 475
        # self.draw.ellipse((home_x - 55, mound_y - 30, home_x + 55, mound_y + 30), fill=dirt_dark, outline=(0,0,0,30), width=3)
        self.draw.rectangle((home_x-8, mound_y-2, home_x+8, mound_y+2), fill=chalk)

        self.draw.ellipse((home_x - 100, home_y - 80, home_x + 100, home_y + 60), fill=dirt)

        # Foul Lines and Batter's Boxes
        self.draw.line((home_x, home_y, 0, 320), fill=chalk, width=5)
        self.draw.line((home_x, home_y, self.width, 320), fill=chalk, width=5)
        self.draw.rectangle((home_x - 80, home_y - 45, home_x - 30, home_y + 45), outline=chalk, width=3)
        self.draw.rectangle((home_x + 30, home_y - 45, home_x + 80, home_y + 45), outline=chalk, width=3)

        # Bases with 3D effect
        coords = self._get_position_coords()
        base_size = 20
        for pos_key in ['1B', '2B_base', '3B']:
            x, y = coords[pos_key]
            # Shadow/3D effect
            self.draw.polygon([(x, y+base_size), (x+base_size, y), (x+base_size, y-base_size), (x, y)], fill=(200,200,200,200))
            # Top of the base
            self.draw.rectangle((x-base_size, y-base_size, x+base_size, y+base_size), fill=(255,255,255,240))
        
        hp_size = 22
        self.draw.polygon([(home_x-hp_size, home_y-hp_size), (home_x+hp_size, home_y-hp_size), (home_x+hp_size, home_y), (home_x, home_y+hp_size/2), (home_x-hp_size, home_y)], fill=chalk)

    def _get_position_coords(self):
        home_x, home_y = self.width / 2, 650
        return {
            "C":  (home_x, home_y + 60), "P":  (home_x, 475),
            # Corrected positions for 1B and 3B to be inside foul lines
            "1B": (590, 510), "2B": (500, 336),
            "SS": (300, 336), "3B": (210, 510),
            "LF": (200, 220), "CF": (home_x, 150), "RF": (self.width - 200, 220),
            "2B_base": (home_x, 350),
        }

    def _get_player_info(self, position):
        player_row = self.lineup_df[self.lineup_df['player_position'] == position]
        if not player_row.empty: return player_row.iloc[0]
        if position == 'P': player_row = self.lineup_df[self.lineup_df['position_group'] == 'Pitchers']
        if position == 'DH': player_row = self.lineup_df[self.lineup_df['position_group'] == 'Designated Hitter']
        return player_row.iloc[0] if not player_row.empty else None

    def _draw_text_with_shadow(self, xy, text, font, fill_color, shadow_color=(0,0,0,180)):
        x, y = xy
        self.draw.text((x + 2, y + 2), text, font=font, fill=shadow_color, anchor="ms")
        self.draw.text((x, y), text, font=font, fill=fill_color, anchor="ms")

    def _draw_player_sprite(self, x, y, player_info):
        jersey_num = str(int(player_info['player_number'])) if player_info is not None and pd.notna(player_info['player_number']) else ""
        
        # Player Shadow
        shadow_box = (x - 20, y + 20, x + 20, y + 30)
        shadow_img = Image.new('RGBA', (40, 10), (0,0,0,0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        shadow_draw.ellipse((0,0,40,10), fill=(0,0,0,80))
        self.image.paste(shadow_img, (int(x-20), int(y+15)), shadow_img)

        # Player Torso, Head, Cap
        self.draw.ellipse((x - 18, y - 25, x + 18, y + 25), fill="#C00000", outline="black", width=2)
        self.draw.ellipse((x - 10, y - 45, x + 10, y - 25), fill="#F0D0B0")
        self.draw.pieslice((x - 12, y - 50, x + 12, y - 30), 180, 360, fill="#A00000")
        
        # Jersey Number
        self._draw_text_with_shadow((x, y + 8), jersey_num, self.font_jersey, "white")

    def _draw_positions(self):
        positions = self._get_position_coords()
        
        for pos_code in ["LF", "RF", "CF", "P", "C", "1B", "2B", "3B", "SS"]: # Draw in layer order
            x, y = positions[pos_code]
            player_info = self._get_player_info(pos_code)
            player_name = player_info['player_name'] if player_info is not None else "Player"
            
            self._draw_player_sprite(x, y, player_info)
            self._draw_text_with_shadow((x, y + 45), pos_code, self.font_pos, "white")
            self._draw_text_with_shadow((x, y + 75), player_name, self.font_player, "white")
            
        dh_info = self._get_player_info('DH')
        if dh_info is not None:
            self._draw_text_with_shadow((self.width / 2, self.height - 40), f"DH: {dh_info['player_name']}", self.font_pos, "white")

    def generate_image_base64(self):
        self._draw_field()
        self._draw_positions()
        
        buffered = io.BytesIO()
        self.image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")