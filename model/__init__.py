def getModelDict(args):
    # if args.eval:
    #     from .NeRV_pavia import NeRV_Generator
    #     from .E_NeRV_pavia_eval import E_NeRV_Generator
    #     model_dict_pavia = {
    #         "NeRV": NeRV_Generator,
    #         "E_NeRV": E_NeRV_Generator,
    #     }
    # else:
    #     from .NeRV_pavia import NeRV_Generator
    #     from .E_NeRV_pavia import E_NeRV_Generator
    #     model_dict_pavia = {
    #         "NeRV": NeRV_Generator,
    #         "E_NeRV": E_NeRV_Generator,
    #     }
    from .NeRV_pavia import NeRV_Generator
    from .E_NeRV_pavia import E_NeRV_Generator
    model_dict_pavia = {
        "NeRV": NeRV_Generator,
        "E_NeRV": E_NeRV_Generator,
    }
    return model_dict_pavia
