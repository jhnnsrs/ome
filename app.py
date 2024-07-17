from typing import List, Optional
import xarray as xr
from arkitekt_next import register
from mikro_next.api.schema import (
    Image,
    from_array_like,
    File,
    Dataset,
    create_instrument,
    create_antibody,
    create_stage,
    Stage,
    PartialRGBViewInput,
    PartialAffineTransformationViewInput,
    PartialOpticsViewInput,
)
import logging
import tifffile
from aicsimageio.metadata.utils import bioformats_ome
from scyjava import config
from aicsimageio import AICSImage
import numpy as np


logger = logging.getLogger(__name__)
x = config


def load_as_xarray(path: str, index: int):
    if path.endswith((".stk", ".tif", ".tiff", ".TIF")):
        image = tifffile.imread(path)
        print(image.shape)

        image = image.reshape((1,) * (5 - image.ndim) + image.shape)
        return xr.DataArray(image, dims=list("ctzyx"))

    else:
        image = AICSImage(path)
        image.set_scene(index)
        image = image.xarray_data.rename(
            {"C": "c", "T": "t", "Z": "z", "X": "x", "Y": "y"}
        )
        image = image.transpose("t", "z", "c", "y", "x")
        image.attrs = {}
        return image


@register()
def convert_omero_file(
    file: File,
    dataset: Optional[Dataset],
    stage: Optional[Stage],
    position_from_planes: bool = True,
    timepoint_from_time: bool = True,
    channels_from_channels: bool = True,
    position_tolerance: Optional[float] = None,
    timepoint_tolerance: Optional[float] = None,
) -> List[Image]:
    """Convert Omero

    Converts an Omero File in a set of Mikrodata

    Args:
        file (OmeroFileFragment): The File to be converted
        stage (Optional[StageFragment], optional): The Stage in which to put the Image. Defaults to None.
        era (Optional[EraFragment], optional): The Era in which to put the Image.. Defaults to None.
        dataset (Optional[DatasetFragment], optional): The Dataset in which to put the Image. Defaults to the file dataset.
        position_from_planes (bool, optional): Whether to create a position from the first planes (only if stage is provided). Defaults to True.
        timepoint_from_time (bool, optional): Whether to create a timepoint from the first time (only if era is provided). Defaults to True.
        channels_from_channels (bool, optional): Whether to create a channel from the channels. Defaults to True.
        position_tolerance (Optional[float], optional): The tolerance for the position. Defaults to no tolerance.
        timepoint_tolerance (Optional[float], optional): The tolerance for the timepoint. Defaults  to no tolerance.

    Returns:
        List[RepresentationFragment]: The created series in this file
    """

    images = []

    assert file.store, "No File Provided"

    f = file.store.download()
    meta = bioformats_ome(f)
    print(meta)
    instrument_map = dict()

    stage = stage or create_stage(f"New Stage for {file.name}")

    for instrument in meta.instruments:
        if instrument.id:
            if instrument.microscope:

                instrument_map[instrument.id] = create_instrument(
                    name=(
                        instrument.microscope.serial_number
                        if instrument.microscope.serial_number
                        else instrument.id
                    ),
                    serial_number=(
                        instrument.microscope.serial_number
                        if instrument.microscope.serial_number
                        else instrument.id
                    ),
                    model=(
                       instrument.microscope.model
                        if instrument.microscope.model
                        else instrument.id
                    ),
                )

    for index, image in enumerate(meta.images):
        # we will create an image for every series here
        print(index)
        pixels = image.pixels
        print(pixels)

        views = []
        # read array (at the moment fake)
        array = load_as_xarray(f, index)
        print(array)

        position = None
        timepoint = None

        transformation_views = []
        optics_views = []

        physical_size_x = pixels.physical_size_x if pixels.physical_size_x else 1
        physical_size_y = pixels.physical_size_y if pixels.physical_size_y else 1
        physical_size_z = pixels.physical_size_z if pixels.physical_size_z else 1


        rgb_views = []



        for index, channel in enumerate(pixels.channels):

            if channel.color:

                value = channel.color.as_rgb_tuple()+ (255,)
                print(value)
                rgb_views.append(
                    PartialRGBViewInput(
                        cMin=index,
                        cMax=index+1,
                        rescale=True,
                        colorMap="INTENSITY",
                        baseColor=value,
                    )
                )




        affine_matrix = np.array(
            [
                [physical_size_x, 0, 0, 0],
                [0, physical_size_y, 0, 0],
                [0, 0, physical_size_z, 0],
                [0, 0, 0, 1],
            ]
        )

        if position_from_planes and len(pixels.planes) > 0:
            first_plane = pixels.planes[0]

            # translate matrix
            affine_matrix[0][3] = first_plane.position_x if first_plane.position_x else 0
            affine_matrix[1][3] = first_plane.position_y if first_plane.position_y else 0
            affine_matrix[2][3] = first_plane.position_z if first_plane.position_z else 0

        afine_matrix = affine_matrix.reshape((4, 4))

        transformation_views.append(
            PartialAffineTransformationViewInput(
                affine_matrix=afine_matrix,
                stage=stage,
            )
        )

        print(instrument_map)

        ins = instrument_map.get(image.instrument_ref.id, None)

        if ins is not None:
            optics_views.append(
                PartialOpticsViewInput(
                    instrument=ins,
                )
            )

        rep = from_array_like(
            array,
            name=file.name + " - " + (image.name if image.name else f"({index})"),
            file_origins=[file],
            tags=["converted"],
            transformation_views=transformation_views,
            optics_views=optics_views,
            rgb_views=rgb_views,
        )


        images.append(rep)

    return images


@register()
def convert_tiff_file(
    file: File,
    dataset: Optional[Dataset],
) -> List[Image]:
    """Convert Tiff File

    Converts an tilffe File in a set of Mikrodata (without metadata)

    Args:
        file (OmeroFileFragment): The File to be converted
        dataset (Optional[DatasetFragment], optional): The dataset that should contain the added images. Defaults to None.

    Returns:
        List[RepresentationFragment]: The created series in this file
    """
    print("images")

    images = []

    assert file.file, "No File Provided"
    with file.file as f:
        image = tifffile.imread(f)

        image = image.reshape((1,) * (5 - image.ndim) + image.shape)
        array = xr.DataArray(image, dims=list("ctzyx"))

        images.append(
            from_array_like(
                array,
                name=file.name,
                datasets=[dataset] if dataset else []   ,
                file_origins=[file],
                tags=["converted"],
            )
        )

    return images
