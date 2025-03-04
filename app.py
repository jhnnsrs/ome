import os
from typing import List, Optional
import xarray as xr
from arkitekt_next import register, easy, progress
from mikro_next.api.schema import (
    Image,
    from_array_like,
    File,
    Dataset,
    create_channel,
    create_instrument,
    create_stage,
    Stage,
    PartialRGBViewInput,
    PartialAffineTransformationViewInput,
    PartialOpticsViewInput,
    PartialScaleViewInput,
    PartialChannelViewInput,
    PartialFileViewInput,
    PartialDerivedViewInput
    
)
from bioio_bioformats.biofile import BioFile
import logging
import tifffile
from scyjava import config
from bioio import BioImage
import numpy as np
from xarray_multiscale import multiscale, windowed_mean



logger = logging.getLogger(__name__)
x = config


def load_as_xarray(image: BioImage, scene: int):
    image.set_scene(scene)


    xarray_dask = image.xarray_dask_data

    if "S" in xarray_dask.dims:
        array = xarray_dask.isel(S=0)
    else:
        array = xarray_dask

    
    image = array.rename(
        {"C": "c", "T": "t", "Z": "z", "X": "x", "Y": "y"}
    )


    image = image.transpose("c", "t", "z", "y", "x")

    x = xr.DataArray(image.data, dims=list("ctzyx"))
    return x


@register(logo="ome.png")
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

    progress(0, "Downloading File")
    f = file.store.download()

    try:
        progress(10, "Downloaded File. Inspecting Metadata")
        aics_image = BioImage(f)
        
        meta = BioFile(f, series=0).ome_metadata
    
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


        amount_images = len(aics_image.scenes)


        start_percent = np.linspace(10, 100, amount_images)




        for index, scene in enumerate(aics_image.scenes):

            image = meta.images[index]

            percent_range = [start_percent[index], start_percent[index+1]] if index+1 < amount_images else [start_percent[index], 100]



            progress(percent_range[0], f"Processing Scene {index+1}/{amount_images}")
            # we will create an image for every series here
            print("The index", index)
            pixels = image.pixels
            print(pixels)

            views = []
            array = load_as_xarray(aics_image, scene)
            print(array)

            position = None
            timepoint = None

            transformation_views = []
            optics_views = []

            physical_size_x = pixels.physical_size_x if pixels.physical_size_x else 1
            physical_size_y = pixels.physical_size_y if pixels.physical_size_y else 1
            physical_size_z = pixels.physical_size_z if pixels.physical_size_z else 1


            rgb_views = []

            channel_views = []

        

        


            for channelindex, channel in enumerate(pixels.channels):

                if channel.color:

                    value = channel.color.as_rgb_tuple()+ (255,)
                    print(value)
                    rgb_views.append(
                        PartialRGBViewInput(
                            cMin=channelindex,
                            cMax=channelindex+1,
                            rescale=True,
                            colorMap="INTENSITY",
                            baseColor=value,
                        )
                    )

                if channel.name:
                    x = create_channel(
                        name=channel.name,
                    )

                    channel_views.append(
                        PartialChannelViewInput(
                            channel=x,
                            cMin=channelindex,
                            cMax=channelindex+1,
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

            if image.instrument_ref:
                ins = instrument_map.get(image.instrument_ref.id, None)

                if ins is not None:
                    optics_views.append(
                        PartialOpticsViewInput(
                            instrument=ins,
                        )
                    )


            array = array.transpose("c", "t", "z", "y", "x")


            progress(percent_range[0], f"Uploading Scene {index+1}/{amount_images}")
            created_image = from_array_like(
                array,
                name=file.name + " - " + (image.name if image.name else f"({index})"),
                tags=["converted"],
                transformation_views=transformation_views,
                optics_views=optics_views,
                channel_views=channel_views,
                rgb_views=rgb_views,
                file_views=[
                    PartialFileViewInput(
                        file=file,
                        seriesIdentifier=image.id,
                    )
                ]
            )

            i = 0
            scale_x = 2
            scale_y = 2
            scale_z = 2 if array.z.size > 5 else 1
            scale_t = 1
            scale_c = 1

            coordless = array.drop_vars(list(array.coords))

            scales = multiscale(
                coordless, windowed_mean, [scale_c, scale_t, scale_z, scale_y, scale_x]
            )

            print(scales)

            

            upload_scales = []


            for i, scale in enumerate(scales):
                
                print(scale.size)
                if scale.shape == array.shape:
                    print("Image the same size")
                    continue


                if scale.size < 20 * 1000 * 1000: # 20 MB
                    print("Image too small")
                    break

                upload_scales.append((i,scale))


            len(upload_scales)

            progress_space = np.linspace(percent_range[0], percent_range[1], len(upload_scales))
            p_i = 0

            for i, scale in upload_scales:

                progress(progress_space[p_i], f"Multiscale: Downscaling {i} for Scene {index+1}/{amount_images}")
                derived_scale = from_array_like(
                    scale,
                    name=f"Scaled of {i}",
                    scale_views=[
                        PartialScaleViewInput(
                            parent=created_image,
                            scaleC=scale_c**i,
                            scaleT=scale_t**i,
                            scaleX=scale_x**i,
                            scaleY=scale_y**i,
                            scaleZ=scale_z**i,
                        )
                    ],
                    derived_views=[
                        PartialDerivedViewInput(
                            originImage=created_image,
                        )
                    ],
                )
                print(derived_scale)
                p_i += 1


            print("Image done")




            images.append(created_image)
    except Exception as e:
        raise e
    
    finally:
        os.remove(f)


    return images


@register
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

@register
def multi_scale(image: Image) -> List[Image]:

    array = image.data
    i = 0
    scale_x = 2
    scale_y = 2
    scale_z = 2 if array.z.size > 5 else 1
    scale_t = 1
    scale_c = 1

    images = []

    coordless = array.drop_vars(list(array.coords))

    scales = multiscale(
        coordless, windowed_mean, [scale_c, scale_t, scale_z, scale_y, scale_x]
    )

    print(scales)

    

    upload_scales = []


    for i, scale in enumerate(scales):
        
        print(scale.size)
        if scale.shape == array.shape:
            print("Image the same size")
            continue


        if scale.size < 1 * 1000 * 1000: # 1 MB
            print("Image too small")
            break

        upload_scales.append((i,scale))


    len(upload_scales)

    progress_space = np.linspace(0, 100, len(upload_scales))
    p_i = 0

    for i, scale in upload_scales:

        progress(progress_space[p_i], f"Multiscale: Downscaling {i}")
        derived_scale = from_array_like(
            scale,
            name=f"Scaled of {i}",
            scale_views=[
                PartialScaleViewInput(
                    parent=image,
                    scaleC=scale_c**i,
                    scaleT=scale_t**i,
                    scaleX=scale_x**i,
                    scaleY=scale_y**i,
                    scaleZ=scale_z**i,
                )
            ],
            derived_views=[
                PartialDerivedViewInput(
                    originImage=image,
                )
            ],
        )
        images.append(derived_scale)
        print(derived_scale)
        p_i += 1


    print("Image done")



    return images



if __name__ == "__main__":
    
    with easy("fuck") as e:

        load_from_file("Breast_Healthy_1_1z.czi")



