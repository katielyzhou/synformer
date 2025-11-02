# ------------------------------------------------------------
# Adapted from AiZynthFinder image.py - credit to them
# ------------------------------------------------------------

import os
import shutil
import tempfile
import atexit
from typing import Dict, Sequence, Tuple

from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw

IMAGE_FOLDER = tempfile.mkdtemp()


@atexit.register
def _clean_up_images() -> None:
    try:
        shutil.rmtree(IMAGE_FOLDER, ignore_errors=True)
    except Exception:
        pass


# ------------------------------------------------------------
# Molecule drawing utilities
# ------------------------------------------------------------
def molecule_to_image(smiles: str, frame_color: str, size: int = 300) -> Image.Image:
    """Create an image of a molecule with a rounded frame."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    img = Draw.MolToImage(mol, size=(size, size))
    cropped = crop_image(img)
    return draw_rounded_rectangle(cropped, frame_color)


def crop_image(img: Image.Image, margin: int = 20) -> Image.Image:
    """Crop whitespace from around a molecule."""
    bg = (255, 255, 255)
    x0, y0 = img.width, img.height
    x1, y1 = 0, 0
    for x in range(img.width):
        for y in range(img.height):
            if img.getpixel((x, y)) != bg:
                x0, y0 = min(x0, x), min(y0, y)
                x1, y1 = max(x1, x), max(y1, y)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1 + 1), min(img.height, y1 + 1)
    cropped = img.crop((x0, y0, x1, y1))
    out = Image.new("RGB", (cropped.width + 2 * margin, cropped.height + 2 * margin), "white")
    out.paste(cropped, (margin+1, margin+1))
    return out


def draw_rounded_rectangle(img: Image.Image, color: str, arc_size: int = 20) -> Image.Image:
    """Draw a rounded rectangle (frame) around an image."""
    x0, y0, x1, y1 = img.getbbox()
    x1 -= 1
    y1 -= 1
    copy = img.copy()
    draw = ImageDraw.Draw(copy)
    arc_half = arc_size // 2

    # Four corners
    draw.arc((x0, y0, arc_size, arc_size), start=180, end=270, fill=color)
    draw.arc((x1 - arc_size, y0, x1, arc_size), start=270, end=0, fill=color)
    draw.arc((x1 - arc_size, y1 - arc_size, x1, y1), start=0, end=90, fill=color)
    draw.arc((x0, y1 - arc_size, arc_size, y1), start=90, end=180, fill=color)

    # Four sides
    draw.line((x0 + arc_half, y0, x1 - arc_half, y0), fill=color)
    draw.line((x1, arc_half, x1, y1 - arc_half), fill=color)
    draw.line((arc_half, y1, x1 - arc_half, y1), fill=color)
    draw.line((x0, arc_half, x0, y1 - arc_half), fill=color)
    return copy


def save_molecule_images(
    smiles_list: Sequence[str],
    frame_colors: Sequence[str],
    size: int = 300,
) -> Dict[str, str]:
    """Create and save molecule images in a temporary directory."""
    spec = {}
    for smiles, frame_color in zip(smiles_list, frame_colors):
        img = molecule_to_image(smiles, frame_color, size)
        inchi_key = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
        filepath = os.path.join(IMAGE_FOLDER, f"{inchi_key}.png")
        img.save(filepath)
        spec[smiles] = filepath
    return spec


# ------------------------------------------------------------
# Route visualization factory
# ------------------------------------------------------------
class RouteImageFactory:
    """
    A lightweight visualizer for retrosynthetic routes using only RDKit and PIL.

    Args:
        route (dict): Dictionary representation of a synthesis route.
        in_stock_colors (dict): Colors for in-stock/out-of-stock molecules.
        show_all (bool): Show hidden nodes if True.
        margin (int): Spacing between nodes.
    """

    def __init__(
        self,
        route: dict,
        in_stock_colors: Dict[bool, str] = None,
        show_all: bool = True,
        margin: int = 100,
    ):
        in_stock_colors = in_stock_colors or {True: "green", False: "orange"}
        self.show_all = show_all
        self.margin = margin

        # Extract molecules
        self._stock_lookup, self._mol_lookup = self._extract_molecules(route)
        self._image_lookup = {
            smi: molecule_to_image(smi, in_stock_colors[self._stock_lookup[smi]])
            for smi in self._mol_lookup.keys()
        }

        # Build route tree
        self._mol_tree = self._extract_mol_tree(route)
        self._add_effective_size(self._mol_tree)

        pos0 = (
            self._mol_tree["eff_width"] - self._mol_tree["image"].width + self.margin,
            int(self._mol_tree["eff_height"] * 0.5) - int(self._mol_tree["image"].height * 0.5),
        )
        self._add_pos(self._mol_tree, pos0)

        # Create final image
        self.image = Image.new(
            "RGB",
            (self._mol_tree["eff_width"] + self.margin, self._mol_tree["eff_height"]),
            color="white",
        )
        self._draw = ImageDraw.Draw(self.image)
        self._make_image(self._mol_tree)
        self.image = crop_image(self.image)

    # ---------------- Internal helpers ---------------- #
    def _extract_molecules(self, tree: dict) -> Tuple[Dict[str, bool], Dict[str, str]]:
        stock_lookup, mol_lookup = {}, {}
        if tree["type"] == "mol":
            stock_lookup[tree["smiles"]] = tree.get("in_stock", False)
            mol_lookup[tree["smiles"]] = tree["smiles"]
        for child in tree.get("children", []):
            child_stock, child_mol = self._extract_molecules(child)
            stock_lookup.update(child_stock)
            mol_lookup.update(child_mol)
        return stock_lookup, mol_lookup

    def _extract_mol_tree(self, tree: dict) -> dict:
        node = {"smiles": tree["smiles"], "image": self._image_lookup[tree["smiles"]]}
        if tree.get("children"):
            node["children"] = [
                self._extract_mol_tree(grandchild)
                for grandchild in tree["children"][0].get("children", [])
                if not (grandchild.get("hide", False) and not self.show_all)
            ]
        return node

    def _add_effective_size(self, node: dict) -> None:
        children = node.get("children", [])
        for child in children:
            self._add_effective_size(child)
        if children:
            node["eff_height"] = sum(c["eff_height"] for c in children) + self.margin * (len(children) - 1)
            node["eff_width"] = max(c["eff_width"] for c in children) + node["image"].width + self.margin
        else:
            node["eff_height"] = node["image"].height + 10 # reactant rectangles can get cut off
            node["eff_width"] = node["image"].width + self.margin

    def _add_pos(self, node: dict, pos: Tuple[int, int]) -> None:
        node["left"], node["top"] = pos
        children = node.get("children", [])
        if not children:
            return
        mid_y = pos[1] + node["image"].height // 2
        total_height = sum(c["eff_height"] for c in children) + self.margin * (len(children) - 1)
        top_y = mid_y - total_height // 2
        for child in children:
            y = top_y + (child["eff_height"] - child["image"].height) // 2
            x = pos[0] - child["image"].width - self.margin
            self._add_pos(child, (x, y))
            top_y += child["eff_height"] + self.margin

    def _make_image(self, node: dict) -> None:
        self.image.paste(node["image"], (node["left"], node["top"]))
        children = node.get("children")
        if not children:
            return

        children_right = max(child["left"] + child["image"].width for child in children)
        mid_x = children_right + int(0.5 * (node["left"] - children_right))
        mid_y = node["top"] + int(node["image"].height * 0.5)

        self._draw.line((node["left"], mid_y, mid_x, mid_y), fill="black")
        for child in children:
            self._make_image(child)
            child_mid_y = child["top"] + int(0.5 * child["image"].height)
            self._draw.line(
                (
                    mid_x,
                    mid_y,
                    mid_x,
                    child_mid_y,
                    child["left"] + child["image"].width,
                    child_mid_y,
                ),
                fill="black",
            )
        self._draw.ellipse(
            (mid_x - 8, mid_y - 8, mid_x + 8, mid_y + 8), fill="black", outline="black"
        )


